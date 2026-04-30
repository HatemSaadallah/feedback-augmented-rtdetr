"""Decoder-to-Encoder feedback for RT-DETR.

After decoder layer 1 makes rough predictions, its query output is fed back to
refine the encoder memory via cross-attention. Subsequent decoder layers
(2..N) then cross-attend over the refined memory.

Two design knobs in v2 (post-v1 ablation showed feedback contributes ~0 at
inference; gate had decayed too low):

    * gate_floor — reparameterize the scalar gate as
          gate_eff = floor + (1 - floor) * sigmoid(alpha)
      so the effective contribution can never drop below `floor`. With
      `floor=0.1, alpha_init=0.0` the gate starts at 0.55 (vs 0.12 in v1
      with `alpha_init=-2.0` and no floor).
    * level_mask — a list[bool] of length num_levels selecting which encoder
      memory levels to refine. When set, cross-attention is computed on the
      subset (saving compute and keeping the attention pattern uncontaminated
      by tokens we are not going to write back to). Unmasked levels pass
      through unchanged.

Shapes (batch B, hidden d=256, input 640x640, 4 levels P2..S5):
    memory  : [B, L, d]  L = sum(H_i * W_i)  e.g. 34000 for P2 input
    dec_out : [B, num_queries, d]
    refined : [B, L, d]   same shape; only active-level positions changed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderToEncoderFeedback(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.0,
                 gate_init: float = 0.0,
                 gate_floor: float = 0.0,
                 level_mask=None):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        assert 0.0 <= gate_floor < 1.0, f'gate_floor must be in [0, 1), got {gate_floor}'
        self.gate_floor = float(gate_floor)

        if level_mask is None:
            self.level_mask = None
        else:
            mask = torch.tensor([bool(x) for x in level_mask], dtype=torch.bool)
            assert mask.any(), 'level_mask must select at least one level'
            self.register_buffer('level_mask', mask, persistent=False)

        # Stashed for visualization. Not in state_dict.
        self.last_attn_weights: torch.Tensor | None = None
        # Toggled by the solver during warmup; when True, forward is a no-op.
        self.disabled: bool = False

    @property
    def effective_gate(self) -> torch.Tensor:
        return self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.gate)

    def _active_indices(self, spatial_shapes, device):
        offset = 0
        chunks = []
        mask_list = self.level_mask.tolist()
        for li, (h_l, w_l) in enumerate(spatial_shapes):
            n_l = int(h_l) * int(w_l)
            if mask_list[li]:
                chunks.append(torch.arange(offset, offset + n_l, device=device))
            offset += n_l
        return torch.cat(chunks) if chunks else None

    def forward(self, memory: torch.Tensor, dec_out: torch.Tensor,
                spatial_shapes=None) -> torch.Tensor:
        if self.disabled:
            self.last_attn_weights = None
            return memory

        if self.level_mask is not None and spatial_shapes is not None:
            active_idx = self._active_indices(spatial_shapes, memory.device)
            if active_idx is None or active_idx.numel() == 0:
                return memory
            sub_mem = memory.index_select(1, active_idx)
        else:
            active_idx = None
            sub_mem = memory

        attn_out, attn_w = self.cross_attn(
            query=sub_mem, key=dec_out, value=dec_out,
            need_weights=True, average_attn_weights=True,
        )
        self.last_attn_weights = attn_w.detach()

        gate = self.effective_gate
        h = self.norm_attn(sub_mem + gate * attn_out)
        h = self.norm_ffn(h + gate * self.ffn(h))

        if active_idx is None:
            return h
        out = memory.clone()
        # Under AMP, `memory` is fp16 but LayerNorm output `h` is fp32;
        # index_copy_ requires matching dtypes, so cast back.
        out.index_copy_(1, active_idx, h.to(memory.dtype))
        return out


class FeedbackAugmentedDecoder(nn.Module):
    """Wraps a TransformerDecoder, injecting memory feedback after layer
    `feedback_layer_idx`. Reuses the base decoder's layers.
    """

    def __init__(self,
                 base_decoder: nn.Module,
                 feedback: DecoderToEncoderFeedback,
                 num_queries: int,
                 feedback_layer_idx: int = 0):
        super().__init__()
        self.layers = base_decoder.layers
        self.num_layers = base_decoder.num_layers
        self.hidden_dim = base_decoder.hidden_dim
        self.eval_idx = base_decoder.eval_idx

        self.feedback = feedback
        self.num_queries = num_queries
        self.feedback_layer_idx = feedback_layer_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        from .utils import inverse_sigmoid  # local import; avoids circular at module load

        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            if i == self.feedback_layer_idx:
                # Only matched (non-denoising) queries are used as K/V so the
                # denoising prefix (train-only) cannot leak GT into memory.
                matched_out = output[:, -self.num_queries:]
                memory = self.feedback(memory, matched_out,
                                       spatial_shapes=memory_spatial_shapes)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                    )
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)

    def set_feedback_disabled(self, disabled: bool) -> None:
        self.feedback.disabled = bool(disabled)

    @property
    def gate_value(self) -> float:
        return float(self.feedback.effective_gate.item())


def build_feedback_module(d_model: int = 256,
                          nhead: int = 8,
                          dim_feedforward: int = 1024,
                          dropout: float = 0.0,
                          gate_init: float = 0.0,
                          gate_floor: float = 0.0,
                          level_mask=None) -> DecoderToEncoderFeedback:
    return DecoderToEncoderFeedback(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        gate_init=gate_init,
        gate_floor=gate_floor,
        level_mask=level_mask,
    )
