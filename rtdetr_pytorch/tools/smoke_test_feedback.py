"""Smoke test: build a tiny RT-DETR with use_feedback on/off, run a dummy
forward pass, verify shapes, param-group membership, gate warmup toggle, and
that attention weights are stashed for visualization.

Runs on CPU with a toy input to keep it fast.
"""

import os
import sys
import re
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Shim: modern torchvision renamed `datapoints` → `tv_tensors`. This repo's
# data module imports `from torchvision import datapoints`, which we don't
# actually hit for our smoke test, but `src/__init__.py` transitively imports
# the data package. Provide a lightweight alias so the import chain resolves.
import torchvision
if not hasattr(torchvision, 'datapoints'):
    try:
        from torchvision import tv_tensors as _tv
        mod = types.ModuleType('torchvision.datapoints')
        mod.BoundingBox = _tv.BoundingBoxes
        mod.BoundingBoxFormat = _tv.BoundingBoxFormat
        mod.Image = _tv.Image
        mod.Mask = _tv.Mask
        mod.Video = _tv.Video
        sys.modules['torchvision.datapoints'] = mod
        torchvision.datapoints = mod
    except Exception:
        pass
if not hasattr(torchvision, 'disable_beta_transforms_warning'):
    torchvision.disable_beta_transforms_warning = lambda: None

# The `src.data` package hard-depends on private torchvision v2 names
# (`ToImageTensor` etc.) that have been renamed. Our smoke test does not use
# data loading, so stub `src.data` with an empty module before importing src.
_stub = types.ModuleType('src.data')
sys.modules['src.data'] = _stub

import torch
import torch.nn as nn

from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.zoo.rtdetr.feedback_module import (
    DecoderToEncoderFeedback,
    FeedbackAugmentedDecoder,
    build_feedback_module,
)


def _dummy_backbone_feats(batch=1, input_size=320, p2: bool = False):
    """Mimic PResNet outputs at the requested pyramid depth.

    Without P2 (default): 3 levels [S3,S4,S5] = strides [8,16,32], channels [512,1024,2048].
    With P2: 4 levels [S2,S3,S4,S5] = strides [4,8,16,32], channels [256,512,1024,2048].
    """
    s = input_size
    if p2:
        return [
            torch.randn(batch, 256,  s // 4,  s // 4),
            torch.randn(batch, 512,  s // 8,  s // 8),
            torch.randn(batch, 1024, s // 16, s // 16),
            torch.randn(batch, 2048, s // 32, s // 32),
        ]
    return [
        torch.randn(batch, 512,  s // 8,  s // 8),
        torch.randn(batch, 1024, s // 16, s // 16),
        torch.randn(batch, 2048, s // 32, s // 32),
    ]


def _build_stack(use_feedback: bool, input_size: int = 320):
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        nhead=8,
        dim_feedforward=1024,
        expansion=1.0,
        depth_mult=1.0,
        eval_spatial_size=None,  # recompute pos_embed each forward so we can use a small input
    )
    transformer = RTDETRTransformer(
        num_classes=80,
        hidden_dim=256,
        num_queries=30,                # small for smoke test
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=3,           # smaller decoder for speed
        dim_feedforward=512,
        num_denoising=0,                # skip DN path (not needed for this smoke test)
        eval_spatial_size=None,
        use_feedback=use_feedback,
        feedback_layer_idx=0,
        feedback_gate_init=-6.0,
        feedback_dim_feedforward=512,
    )
    return encoder, transformer


def test_shapes_and_forward():
    torch.manual_seed(0)
    feats = _dummy_backbone_feats(batch=1, input_size=320)

    for use_feedback in (False, True):
        encoder, transformer = _build_stack(use_feedback=use_feedback)
        encoder.eval(); transformer.eval()
        with torch.no_grad():
            enc_out = encoder(feats)
            assert len(enc_out) == 3
            assert enc_out[0].shape == (1, 256, 40, 40), f'S3 got {enc_out[0].shape}'
            assert enc_out[1].shape == (1, 256, 20, 20), f'S4 got {enc_out[1].shape}'
            assert enc_out[2].shape == (1, 256, 10, 10), f'S5 got {enc_out[2].shape}'

            out = transformer(enc_out)
        assert 'pred_logits' in out and 'pred_boxes' in out
        assert out['pred_logits'].shape == (1, 30, 80), f'logits {out["pred_logits"].shape}'
        assert out['pred_boxes'].shape  == (1, 30, 4),  f'boxes {out["pred_boxes"].shape}'
        print(f'  use_feedback={use_feedback}: forward OK, shapes OK')

        if use_feedback:
            # Wrapper type check
            assert isinstance(transformer.decoder, FeedbackAugmentedDecoder)
            # Attention weights stashed for visualization.
            # Multi-level: L = H3*W3 + H4*W4 + H5*W5. For 320 input and
            # strides [8,16,32]: 40*40 + 20*20 + 10*10 = 1600+400+100 = 2100.
            aw = transformer.decoder.feedback.last_attn_weights
            assert aw is not None, 'feedback did not stash attn weights'
            expected_L = 40 * 40 + 20 * 20 + 10 * 10
            assert aw.shape == (1, expected_L, 30), f'attn_w {aw.shape}'
            print(f'  attn_weights stashed: shape={tuple(aw.shape)} '
                  f'(S3:{40*40} + S4:{20*20} + S5:{10*10} = {expected_L})')


def test_warmup_toggle_disables_feedback():
    torch.manual_seed(0)
    feats = _dummy_backbone_feats(batch=1, input_size=320)
    encoder, transformer = _build_stack(use_feedback=True)
    encoder.eval(); transformer.eval()
    with torch.no_grad():
        enc_out = encoder(feats)

        # With warmup ON, feedback is a no-op; the S5 slice of memory must be
        # unchanged by the wrapper (and no attn weights should be stashed).
        transformer.decoder.set_feedback_disabled(True)
        out_disabled = transformer(enc_out)
        assert transformer.decoder.feedback.last_attn_weights is None

        # With warmup OFF, attn weights should be populated.
        transformer.decoder.set_feedback_disabled(False)
        _ = transformer(enc_out)
        assert transformer.decoder.feedback.last_attn_weights is not None
    print('  warmup toggle OK (disabled=True → no attn; False → attn stashed)')


def test_optim_param_groups():
    """Mimic YAMLConfig.get_optim_params exactly with the regexes we use in
    the feedback YAML; verify every parameter lands in exactly one group.
    """
    from src.zoo.rtdetr.rtdetr import RTDETR
    from src.nn.backbone.presnet import PResNet
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256,
        use_encoder_idx=[2], num_encoder_layers=1, eval_spatial_size=None,
    )
    transformer = RTDETRTransformer(
        num_classes=80, hidden_dim=256, num_queries=30,
        feat_channels=[256, 256, 256], feat_strides=[8, 16, 32],
        num_levels=3, num_decoder_layers=2, num_denoising=0,
        dim_feedforward=512, eval_spatial_size=None,
        use_feedback=True,
    )
    # PResNet requires weights download on first use; use a stub with similar
    # param naming to keep the test offline.
    class StubBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, 2, 1)
            self.norm = nn.BatchNorm2d(64)
    model = RTDETR(backbone=StubBackbone(), encoder=encoder, decoder=transformer)

    patterns = [
        'backbone',
        r'decoder\.feedback',
        r'^(?=.*encoder(?=.*bias|.*norm.*weight)).*$',
        r'^(?!.*feedback)(?=.*decoder(?=.*bias|.*norm.*weight)).*$',
    ]
    names = [k for k, v in model.named_parameters() if v.requires_grad]
    visited = []
    per_group = []
    for p in patterns:
        matched = [k for k in names if len(re.findall(p, k)) > 0]
        per_group.append((p, matched))
        visited.extend(matched)

    dup = set([n for n in visited if visited.count(n) > 1])
    unseen = set(names) - set(visited)
    print(f'  total params={len(names)} matched={len(visited)} unique={len(set(visited))}')
    for p, matched in per_group:
        print(f"    pattern {p!r:60s} → {len(matched):4d} params")
    print(f'  unmatched (catch-all group): {len(unseen)} params')
    assert not dup, f'overlapping pattern match (YAML get_optim_params would assert): {sorted(dup)[:5]}'
    feedback_group_names = per_group[1][1]
    assert all('decoder.feedback.' in n for n in feedback_group_names)
    assert any('gate' in n for n in feedback_group_names), 'gate parameter must be in feedback group'
    print('  all patterns disjoint, feedback params isolated, gate param present')


def test_backward_and_gate_grad():
    torch.manual_seed(0)
    feats = _dummy_backbone_feats(batch=1, input_size=320)
    encoder, transformer = _build_stack(use_feedback=True)
    encoder.train(); transformer.train()

    # Warmup-off path: feedback engages, gate must receive a non-zero grad.
    transformer.decoder.set_feedback_disabled(False)
    enc_out = encoder(feats)
    out = transformer(enc_out)
    loss = out['pred_logits'].abs().mean() + out['pred_boxes'].abs().mean()
    loss.backward()
    g = transformer.decoder.feedback.gate.grad
    assert g is not None and torch.isfinite(g).all()
    print(f'  gate grad (warmup off): {g.item():+.3e}')

    # Warmup-on path: feedback is a no-op; gate should NOT receive gradient.
    for p in transformer.parameters():
        if p.grad is not None:
            p.grad = None
    transformer.decoder.set_feedback_disabled(True)
    enc_out = encoder(feats)
    out = transformer(enc_out)
    loss = out['pred_logits'].abs().mean() + out['pred_boxes'].abs().mean()
    loss.backward()
    assert transformer.decoder.feedback.gate.grad is None
    print('  gate grad (warmup on): None (as expected)')


def test_end_to_end_with_criterion():
    """Full integration: fake COCO-style batch → RTDETR(feedback) → SetCriterion
    → backward. Verifies the feedback model is compatible with the real loss.
    """
    from src.zoo.rtdetr.matcher import HungarianMatcher
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion

    torch.manual_seed(0)
    feats = _dummy_backbone_feats(batch=2, input_size=320)
    encoder, transformer = _build_stack(use_feedback=True)
    encoder.train(); transformer.train()
    transformer.decoder.set_feedback_disabled(False)

    # Real criterion matching the COCO config
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        use_focal_loss=True, alpha=0.25, gamma=2.0,
    )
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
        losses=['vfl', 'boxes'],
        alpha=0.75, gamma=2.0, num_classes=80,
    )

    # Fake per-image targets (COCO style: labels + normalized cxcywh boxes).
    # Image 0: 2 objects. Image 1: 1 object.
    targets = [
        {
            'labels': torch.tensor([3, 17], dtype=torch.long),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3],
                                   [0.3, 0.7, 0.15, 0.15]], dtype=torch.float32),
        },
        {
            'labels': torch.tensor([42], dtype=torch.long),
            'boxes': torch.tensor([[0.4, 0.4, 0.2, 0.2]], dtype=torch.float32),
        },
    ]

    enc_out = encoder(feats)
    outputs = transformer(enc_out, targets=targets)
    loss_dict = criterion(outputs, targets)
    loss = sum(loss_dict.values())
    assert torch.isfinite(loss), f'loss not finite: {loss.item()}, components={loss_dict}'
    loss.backward()

    gate = transformer.decoder.feedback.gate
    assert gate.grad is not None and torch.isfinite(gate.grad)
    fb_params = [(n, p) for n, p in transformer.named_parameters() if 'feedback' in n]
    with_grad = sum(1 for _, p in fb_params if p.grad is not None)
    print(f'  total loss: {loss.item():.4f}  components: '
          f'{ {k: round(v.item(), 4) for k, v in loss_dict.items()} }')
    print(f'  feedback params: {len(fb_params)} total, {with_grad} received gradient')
    print(f'  gate grad: {gate.grad.item():+.3e}  gate value: {torch.sigmoid(gate).item():.4f}')
    assert with_grad == len(fb_params), 'some feedback params did not receive a gradient'


def test_p2_four_level_forward():
    """Verify the 4-level (S2+S3+S4+S5) architecture works end-to-end.

    At input 320, levels are:
        S2: 80x80 = 6400 tokens
        S3: 40x40 = 1600 tokens
        S4: 20x20 =  400 tokens
        S5: 10x10 =  100 tokens
    Total memory L = 8500 tokens.
    """
    torch.manual_seed(0)
    feats = _dummy_backbone_feats(batch=1, input_size=320, p2=True)
    encoder = HybridEncoder(
        in_channels=[256, 512, 1024, 2048],
        feat_strides=[4, 8, 16, 32],
        use_encoder_idx=[3],
        hidden_dim=256, nhead=8, dim_feedforward=512, num_encoder_layers=1,
        eval_spatial_size=None,
    )
    transformer = RTDETRTransformer(
        num_classes=80, hidden_dim=256, num_queries=50,
        feat_channels=[256, 256, 256, 256], feat_strides=[4, 8, 16, 32],
        num_levels=4, num_decoder_points=4, nhead=8,
        num_decoder_layers=3, dim_feedforward=512,
        num_denoising=0, eval_spatial_size=None,
        use_feedback=True, feedback_layer_idx=0,
        feedback_gate_init=-2.0, feedback_dim_feedforward=512,
    )
    encoder.eval(); transformer.eval()
    with torch.no_grad():
        enc_out = encoder(feats)
        assert len(enc_out) == 4, f'expected 4 encoder outputs, got {len(enc_out)}'
        assert enc_out[0].shape == (1, 256, 80, 80)  # S2
        assert enc_out[1].shape == (1, 256, 40, 40)  # S3
        assert enc_out[2].shape == (1, 256, 20, 20)  # S4
        assert enc_out[3].shape == (1, 256, 10, 10)  # S5
        out = transformer(enc_out)
    assert out['pred_logits'].shape == (1, 50, 80)
    assert out['pred_boxes'].shape == (1, 50, 4)
    aw = transformer.decoder.feedback.last_attn_weights
    expected_L = 80*80 + 40*40 + 20*20 + 10*10  # 8500
    assert aw.shape == (1, expected_L, 50), f'attn_w {aw.shape}'
    print(f'  4-level forward OK: enc shapes [80,40,20,10], memory={expected_L} tokens')


def main():
    print('1) shapes + forward (use_feedback=False, True)')
    test_shapes_and_forward()
    print('2) warmup toggle')
    test_warmup_toggle_disables_feedback()
    print('3) optimizer param-group regex disjointness')
    test_optim_param_groups()
    print('4) backward + gate gradient')
    test_backward_and_gate_grad()
    print('5) end-to-end with SetCriterion (fake COCO-style batch)')
    test_end_to_end_with_criterion()
    print('6) P2 (4-level) architecture forward')
    test_p2_four_level_forward()
    print('\nALL SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
