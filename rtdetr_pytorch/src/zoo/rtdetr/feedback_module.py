"""Decoder-to-Encoder feedback for RT-DETR.

After decoder layer 1 makes rough predictions, its query output is fed back to
refine the S5 slice of the encoder memory via cross-attention. Subsequent
decoder layers (2..N) then cross-attend over the refined memory.

Shapes (batch size B, hidden dim d=256, input 640x640):
    s5_tokens : [B, H5*W5, d]   e.g. [B, 400, 256] for 20x20 S5
    dec_out   : [B, num_queries, d]   e.g. [B, 300, 256]
    refined   : [B, H5*W5, d]

The module has a learnable scalar gate (init ≈ -6 so sigmoid(-6) ≈ 0.0025):
feedback starts effectively disabled and opens as training progresses. A
`disabled` flag (toggled externally by the solver during epoch warmup) skips
the feedback entirely for the first few epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import inverse_sigmoid


class DecoderToEncoderFeedback(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.0,
                 gate_init: float = -6.0):
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

        # Filled in forward; consumed by visualization. Not in state_dict.
        self.last_attn_weights: torch.Tensor | None = None

        # Toggled by the solver during warmup; when True, forward is a no-op.
        self.disabled: bool = False

    def forward(self, s5_tokens: torch.Tensor, dec_out: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            self.last_attn_weights = None
            return s5_tokens

        attn_out, attn_w = self.cross_attn(
            query=s5_tokens, key=dec_out, value=dec_out,
            need_weights=True, average_attn_weights=True,
        )
        # Store for visualization (detached, no graph).
        self.last_attn_weights = attn_w.detach()

        gate = torch.sigmoid(self.gate)
        h = self.norm_attn(s5_tokens + gate * attn_out)
        h = self.norm_ffn(h + gate * self.ffn(h))
        return h


class FeedbackAugmentedDecoder(nn.Module):
    """Wraps a TransformerDecoder, injecting S5 feedback after layer `feedback_layer_idx`.

    Reuses the base decoder's layers (shared parameters). Mirrors the base
    decoder's forward signature so it is a drop-in replacement.
    """

    def __init__(self,
                 base_decoder: nn.Module,
                 feedback: DecoderToEncoderFeedback,
                 num_queries: int,
                 feedback_layer_idx: int = 0,
                 s5_level_idx: int = -1):
        super().__init__()
        self.layers = base_decoder.layers
        self.num_layers = base_decoder.num_layers
        self.hidden_dim = base_decoder.hidden_dim
        self.eval_idx = base_decoder.eval_idx

        self.feedback = feedback
        self.num_queries = num_queries
        self.feedback_layer_idx = feedback_layer_idx
        self.s5_level_idx = s5_level_idx

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
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        # Derive the S5 slice of `memory` from the spatial shapes. The last
        # level (index -1) is S5 in the RT-DETR convention (strides 8,16,32).
        s5_idx = (self.s5_level_idx
                  if self.s5_level_idx >= 0
                  else len(memory_spatial_shapes) + self.s5_level_idx)
        s5_start = memory_level_start_index[s5_idx]
        h5, w5 = memory_spatial_shapes[s5_idx]
        s5_end = s5_start + h5 * w5

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            if i == self.feedback_layer_idx:
                # Use only the matched (non-denoising) query slice as K/V so
                # the denoising prefix (training only) cannot leak GT info.
                matched_out = output[:, -self.num_queries:]
                s5_tokens = memory[:, s5_start:s5_end]
                refined = self.feedback(s5_tokens, matched_out)
                # Splice refined S5 back into full memory; preserve the S3/S4
                # prefix so subsequent deformable attention uses updated S5.
                memory = torch.cat(
                    [memory[:, :s5_start], refined, memory[:, s5_end:]], dim=1,
                )

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
        return float(torch.sigmoid(self.feedback.gate).item())


def build_feedback_module(d_model: int = 256,
                          nhead: int = 8,
                          dim_feedforward: int = 1024,
                          dropout: float = 0.0,
                          gate_init: float = -6.0) -> DecoderToEncoderFeedback:
    return DecoderToEncoderFeedback(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        gate_init=gate_init,
    )
