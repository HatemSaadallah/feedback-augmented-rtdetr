"""Decoder-to-Encoder feedback for RT-DETR.

After decoder layer 1 makes rough predictions, its query output is fed back to
refine the full multi-level encoder memory (S3+S4+S5) via cross-attention.
Subsequent decoder layers (2..N) then cross-attend over the refined memory.

This differs from the original proposal (S5 only) because RT-DETR's
deformable decoder looks at all three levels, and small objects (< 32 px)
live in S3 (stride 8) — not S5 (stride 32, one cell covers 32x32 px). To
help small objects we must refine S3/S4 too.

Shapes (batch size B, hidden dim d=256, input 640x640):
    memory  : [B, L, d]   L = H3*W3 + H4*W4 + H5*W5 (e.g. 8400)
    dec_out : [B, num_queries, d]   e.g. [B, 300, 256]
    refined : [B, L, d]

The module has a learnable scalar gate. Default init is `-2.0` so
`sigmoid(-2) ≈ 0.12` — feedback starts meaningful but non-dominant, and
actually escapes the saturation region during a realistic training schedule.
(With init `-6` the gate took effectively forever to open.) The `disabled`
flag is toggled externally by the solver during epoch warmup and fully
skips the feedback.
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
                 gate_init: float = -2.0):
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

        # Stashed for visualization. Not in state_dict.
        self.last_attn_weights: torch.Tensor | None = None
        # Toggled by the solver during warmup; when True, forward is a no-op.
        self.disabled: bool = False

    def forward(self, memory: torch.Tensor, dec_out: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            self.last_attn_weights = None
            return memory

        attn_out, attn_w = self.cross_attn(
            query=memory, key=dec_out, value=dec_out,
            need_weights=True, average_attn_weights=True,
        )
        self.last_attn_weights = attn_w.detach()

        gate = torch.sigmoid(self.gate)
        h = self.norm_attn(memory + gate * attn_out)
        h = self.norm_ffn(h + gate * self.ffn(h))
        return h


class FeedbackAugmentedDecoder(nn.Module):
    """Wraps a TransformerDecoder, injecting multi-level memory feedback
    after layer `feedback_layer_idx`. Reuses the base decoder's layers.
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
                # Use only matched (non-denoising) queries as K/V so the
                # denoising prefix (train-only) cannot leak GT into memory.
                matched_out = output[:, -self.num_queries:]
                # Refine the *full* multi-level memory — S3+S4+S5. This is
                # the key change vs. the S5-only variant: it gives feedback
                # direct access to the S3 tokens where small objects live.
                memory = self.feedback(memory, matched_out)

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
                          gate_init: float = -2.0) -> DecoderToEncoderFeedback:
    return DecoderToEncoderFeedback(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        gate_init=gate_init,
    )
