"""Clean, professional 18-slide TA presentation.

Run with:
    /home/hatem/miniconda3/envs/rtdetr/bin/python build_pptx_clean.py

Outputs:
    feedback_rtdetr_presentation.pptx     (the deliverable)
    _build/*.png                          (all rendered diagrams + math)

Design rules (per spec):
    - White background, navy / green / red palette only
    - Max 20–25 words per slide
    - Math slides: clean equation with labeled arrows pointing to variables
    - Big numbers for headline results (80–120pt)
    - Diagrams drawn in matplotlib so arrows land exactly where they should
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import rcParams

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


ROOT = Path(__file__).parent
BUILD = ROOT / "_build"
BUILD.mkdir(exist_ok=True)
FIG_DIR = ROOT.parent / "report" / "figures"
V2_DIR = ROOT.parent / "v2_results"

OUT = ROOT / "feedback_rtdetr_presentation.pptx"

# ---- palette (navy + green + red on white) ----
NAVY  = RGBColor(0x1F, 0x6F, 0xEB)
DARK  = RGBColor(0x1A, 0x1A, 0x2E)
LIGHT = RGBColor(0x66, 0x66, 0x77)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
RED   = RGBColor(0xD6, 0x2C, 0x2E)
BG    = RGBColor(0xFF, 0xFF, 0xFF)

NAVY_HEX  = "#1F6FEB"
DARK_HEX  = "#1A1A2E"
LIGHT_HEX = "#66667A"
GREEN_HEX = "#2CA02C"
RED_HEX   = "#D62C2E"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ===========================================================================
# Diagram + math rendering (matplotlib, transparent PNGs)
# ===========================================================================
def _save(fig, name):
    out = BUILD / name
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05,
                transparent=True)
    plt.close(fig)
    return out


def fig_baseline_pipeline():
    """Image → Encoder → Decoder → Predictions, single-pass."""
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.4)
    # 4 boxes — each box has its own (x, width); positions chosen so the
    # full diagram (including the rightmost box) fits inside [0, 11].
    boxes = [
        # (label,   x,    width)
        ("Image",   0.4, 1.6),
        ("Encoder", 2.7, 2.0),
        ("Decoder", 5.4, 2.0),
        ("Predictions", 8.1, 2.5),
    ]
    for label, x, w in boxes:
        rect = FancyBboxPatch((x, 0.7), w, 1.0, boxstyle="round,pad=0.05",
                              linewidth=2, edgecolor=NAVY_HEX, facecolor="white")
        ax.add_patch(rect)
        ax.text(x + w/2, 1.2, label, fontsize=20, ha="center", va="center",
                color=DARK_HEX, fontweight="bold")
    # arrows between consecutive boxes
    for (l1, x1, w1), (l2, x2, w2) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(x2, 1.2), xytext=(x1 + w1, 1.2),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color=DARK_HEX))
    ax.text(5.5, 0.25, "one-way information flow",
            fontsize=14, ha="center", va="center",
            color=LIGHT_HEX, style="italic")
    return _save(fig, "diag_baseline.png")


def fig_idea_pipeline():
    """Encoder ↔ Decoder with feedback loop."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.2)
    # 2 boxes
    enc = FancyBboxPatch((2.6, 1.2), 2.4, 1.0, boxstyle="round,pad=0.05",
                          linewidth=2.5, edgecolor=NAVY_HEX, facecolor="white")
    dec = FancyBboxPatch((6.4, 1.2), 2.4, 1.0, boxstyle="round,pad=0.05",
                          linewidth=2.5, edgecolor=NAVY_HEX, facecolor="white")
    ax.add_patch(enc); ax.add_patch(dec)
    ax.text(3.8, 1.7, "Encoder", fontsize=22, ha="center", va="center",
            color=DARK_HEX, fontweight="bold")
    ax.text(7.6, 1.7, "Decoder", fontsize=22, ha="center", va="center",
            color=DARK_HEX, fontweight="bold")
    # forward arrow (top)
    ax.annotate("", xy=(6.3, 1.85), xytext=(5.0, 1.85),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=DARK_HEX))
    ax.text(5.65, 2.1, "memory", fontsize=12, ha="center", color=LIGHT_HEX)
    # feedback arrow (bottom, curved)
    arr = FancyArrowPatch((6.3, 1.55), (5.0, 1.55),
                          connectionstyle="arc3,rad=-0.5",
                          arrowstyle="->", lw=2.5, color=NAVY_HEX)
    ax.add_patch(arr)
    ax.text(5.65, 0.65, "feedback", fontsize=14, ha="center",
            color=NAVY_HEX, fontweight="bold")
    return _save(fig, "diag_idea.png")


def fig_attention_math():
    """Standard MHA equation with arrows to Q, K, V."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6)

    # main equation
    ax.text(6.5, 3.0,
            r"$\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \;=\; "
            r"\mathrm{softmax}\!\left(\frac{\mathbf{Q}\,\mathbf{K}^{\top}}"
            r"{\sqrt{d}}\right)\mathbf{V}$",
            fontsize=46, ha="center", va="center", color=DARK_HEX)

    # arrows + labels (Q, K, V)
    annotations = [
        # (target_x, target_y, text, label_x, label_y)
        (3.05, 3.15, "what we\nsearch for", 1.6, 5.0),
        (3.55, 3.15, "where we\nlook",       4.7, 5.0),
        (4.05, 3.15, "information\nwe extract", 7.7, 5.0),
    ]
    for i, (tx, ty, label, lx, ly) in enumerate(annotations):
        # we have three labels for Q, K, V — they all sit on top of the equation
        # (Q is at ~3.0, K at ~3.5, V at ~4.0 in the unscaled coords above —
        # but layout depends on font width; instead, use the macro pattern of
        # pointing at the (Q,K,V) tuple and the V at the end.
        pass

    # Use one-shot manual placements that match the rendered equation visually:
    # mathtext bbox is tight; the (Q,K,V) tuple appears around x=3.7..4.7,
    # and the trailing V appears around x=9.2.
    arrows = [
        # Q — left of the tuple
        dict(label="Q\nwhat we search for", lx=2.6, ly=4.9, tx=3.95, ty=3.30),
        # K — middle of the tuple
        dict(label="K\nwhere we look",       lx=5.4, ly=4.9, tx=4.55, ty=3.30),
        # V — right of the tuple (and V at the end)
        dict(label="V\ninformation we extract", lx=8.7, ly=4.9, tx=5.20, ty=3.30),
    ]
    for a in arrows:
        ax.annotate(
            a["label"],
            xy=(a["tx"], a["ty"]),
            xytext=(a["lx"], a["ly"]),
            fontsize=15, ha="center", va="bottom",
            color=NAVY_HEX, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NAVY_HEX, lw=1.6,
                            connectionstyle="arc3,rad=-0.2"),
        )

    # bottom caption
    ax.text(6.5, 0.9,
            "Attention lets the model focus on important parts of the image.",
            fontsize=18, ha="center", va="center",
            color=DARK_HEX, style="italic")
    return _save(fig, "math_attention.png")


def fig_feedback_math():
    """Feedback formula m' = m + g · Attn(m, t) with arrows to m, t, g."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6)

    # main equation
    ax.text(6.5, 3.0,
            r"$\mathbf{m}' \;=\; \mathbf{m} \;+\; \mathbf{g} \cdot "
            r"\mathrm{Attn}(\mathbf{m},\, \mathbf{t})$",
            fontsize=52, ha="center", va="center", color=DARK_HEX)

    # arrows
    annots = [
        # m on the left
        dict(label="m\nencoder memory\n(features)",
             lx=2.4, ly=4.9, tx=4.55, ty=3.25),
        # g
        dict(label="g\ngate\n(how much feedback)",
             lx=6.4, ly=4.9, tx=6.55, ty=3.25),
        # t
        dict(label="t\ndecoder output\n(predictions)",
             lx=10.4, ly=4.9, tx=9.05, ty=3.25),
    ]
    for a in annots:
        ax.annotate(
            a["label"],
            xy=(a["tx"], a["ty"]),
            xytext=(a["lx"], a["ly"]),
            fontsize=15, ha="center", va="bottom",
            color=NAVY_HEX, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NAVY_HEX, lw=1.6,
                            connectionstyle="arc3,rad=-0.2"),
        )

    ax.text(6.5, 0.9,
            "We update the encoder using the decoder's own predictions.",
            fontsize=18, ha="center", va="center",
            color=DARK_HEX, style="italic")
    return _save(fig, "math_feedback.png")


def fig_v2_fixes():
    """Two boxes: gate floor + P2/P3 mask."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
    # box 1
    b1 = FancyBboxPatch((0.4, 0.6), 4.8, 2.4, boxstyle="round,pad=0.1",
                        linewidth=2.5, edgecolor=NAVY_HEX, facecolor="#F0F6FF")
    ax.add_patch(b1)
    ax.text(2.8, 2.55, "1.  Gate floor", fontsize=22, ha="center", va="center",
            color=NAVY_HEX, fontweight="bold")
    ax.text(2.8, 1.85, r"$g_\mathrm{eff} = 0.1 + 0.9\,\sigma(\alpha)$",
            fontsize=18, ha="center", va="center", color=DARK_HEX)
    ax.text(2.8, 1.05, "feedback can never be silenced",
            fontsize=14, ha="center", va="center", color=LIGHT_HEX, style="italic")
    # box 2
    b2 = FancyBboxPatch((5.8, 0.6), 4.8, 2.4, boxstyle="round,pad=0.1",
                        linewidth=2.5, edgecolor=GREEN_HEX, facecolor="#F0FFF0")
    ax.add_patch(b2)
    ax.text(8.2, 2.55, "2.  P2 / P3 only", fontsize=22, ha="center", va="center",
            color=GREEN_HEX, fontweight="bold")
    ax.text(8.2, 1.85, "refine where small objects live",
            fontsize=16, ha="center", va="center", color=DARK_HEX)
    ax.text(8.2, 1.05, "skip P4, P5 — focus the gradient",
            fontsize=14, ha="center", va="center", color=LIGHT_HEX, style="italic")
    return _save(fig, "diag_v2_fixes.png")


def fig_ablation_explainer():
    """Linear flow: checkpoint → 2 evals (stacked) → Δ result."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2)

    # left: the checkpoint
    cp = FancyBboxPatch((0.6, 1.6), 2.6, 1.0, boxstyle="round,pad=0.05",
                        linewidth=2.5, edgecolor=DARK_HEX, facecolor="white")
    ax.add_patch(cp)
    ax.text(1.9, 2.1, "v2 checkpoint", fontsize=18, ha="center", va="center",
            color=DARK_HEX, fontweight="bold")

    # arrows splitting up + down
    ax.annotate("", xy=(5.4, 3.3), xytext=(3.2, 2.4),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=GREEN_HEX))
    ax.annotate("", xy=(5.4, 0.9), xytext=(3.2, 1.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=RED_HEX))

    # two eval nodes
    on_box = FancyBboxPatch((5.4, 2.85), 3.4, 1.0, boxstyle="round,pad=0.05",
                             linewidth=2.5, edgecolor=GREEN_HEX,
                             facecolor="#E8F7E8")
    ax.add_patch(on_box)
    ax.text(7.1, 3.35, "evaluate · feedback ON", fontsize=15, ha="center",
            va="center", color=GREEN_HEX, fontweight="bold")

    off_box = FancyBboxPatch((5.4, 0.45), 3.4, 1.0, boxstyle="round,pad=0.05",
                              linewidth=2.5, edgecolor=RED_HEX,
                              facecolor="#FDECEC")
    ax.add_patch(off_box)
    ax.text(7.1, 0.95, "evaluate · feedback OFF", fontsize=15, ha="center",
            va="center", color=RED_HEX, fontweight="bold")

    # arrows merging into Δ
    ax.annotate("", xy=(11.2, 2.4), xytext=(8.8, 3.35),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=GREEN_HEX))
    ax.annotate("", xy=(11.2, 1.8), xytext=(8.8, 0.95),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=RED_HEX))

    # delta result on the right
    delta_box = FancyBboxPatch((11.0, 1.6), 1.6, 1.0, boxstyle="round,pad=0.05",
                                linewidth=2.5, edgecolor=NAVY_HEX,
                                facecolor="#F0F6FF")
    ax.add_patch(delta_box)
    ax.text(11.8, 2.1, "Δ", fontsize=32, ha="center", va="center",
            color=NAVY_HEX, fontweight="bold")
    return _save(fig, "diag_ablation.png")


def render_all_figures():
    rcParams["mathtext.fontset"] = "cm"
    print("Rendering matplotlib diagrams + math...")
    fig_baseline_pipeline()
    fig_idea_pipeline()
    fig_attention_math()
    fig_feedback_math()
    fig_v2_fixes()
    fig_ablation_explainer()
    print(f"  → {len(list(BUILD.glob('*.png')))} PNGs in {BUILD}")


# ===========================================================================
# pptx helpers
# ===========================================================================
def add_blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_text(slide, x, y, w, h, text, *,
             size=18, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             font="Helvetica", italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def add_title(slide, text, *, color=NAVY):
    add_text(slide, Inches(0.7), Inches(0.5), Inches(12.0), Inches(0.9),
             text, size=34, bold=True, color=color)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.7), Inches(1.25), Inches(0.7), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()


def add_image(slide, path, x, y, w=None, h=None):
    if w and h:
        return slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    if w:
        return slide.shapes.add_picture(str(path), x, y, width=w)
    if h:
        return slide.shapes.add_picture(str(path), x, y, height=h)
    return slide.shapes.add_picture(str(path), x, y)


def add_footer(slide, idx, total):
    add_text(slide, Inches(0.7), Inches(7.05), Inches(8.0), Inches(0.3),
             "Feedback-Augmented RT-DETR  •  Hatem Saadallah & Nour Jennane",
             size=10, color=LIGHT)
    add_text(slide, Inches(11.5), Inches(7.05), Inches(1.2), Inches(0.3),
             f"{idx} / {total}", size=10, color=LIGHT, align=PP_ALIGN.RIGHT)


# ===========================================================================
# Build the deck (18 slides)
# ===========================================================================
def build():
    on  = json.load(open(V2_DIR / "ablations" / "ablation_v2_feedback_on_640.json"))
    off = json.load(open(V2_DIR / "ablations" / "ablation_v2_feedback_off_640.json"))
    on800 = json.load(open(V2_DIR / "ablations" / "ablation_v2_feedback_on_800.json"))
    APS_ON  = on["AP_S"]  * 100
    APS_OFF = off["AP_S"] * 100
    APS_800 = on800["AP_S"] * 100
    DELTA = APS_ON - APS_OFF

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    TOTAL = 18

    # ---------- 1. Title ----------
    s = add_blank_slide(prs)
    add_text(s, Inches(0.8), Inches(2.1), Inches(11.7), Inches(0.5),
             "BOCCONI UNIVERSITY  ·  COMPUTER VISION & IMAGE PROCESSING",
             size=12, color=NAVY, bold=True)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                              Inches(0.8), Inches(2.65), Inches(2.5), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    # Title — single line at 50pt (fits within 11.7" wide frame)
    add_text(s, Inches(0.8), Inches(2.85), Inches(11.7), Inches(1.1),
             "Feedback-Augmented RT-DETR",
             size=50, bold=True, color=DARK)
    # Subtitle on its own row, well below the title
    add_text(s, Inches(0.8), Inches(4.05), Inches(11.7), Inches(0.7),
             "for Small Object Detection",
             size=28, color=LIGHT)
    # Authors + date
    add_text(s, Inches(0.8), Inches(5.7), Inches(11.7), Inches(0.5),
             "Hatem Saadallah  ·  Nour Jennane",
             size=18, color=DARK)
    add_text(s, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.4),
             "Final Project  ·  April 2026",
             size=14, color=LIGHT)

    # ---------- 2. Problem ----------
    s = add_blank_slide(prs)
    add_title(s, "Small objects are harder to detect")
    # left: viz
    add_image(s, FIG_DIR / "viz_000000001000.png",
              Inches(0.7), Inches(1.7), w=Inches(7.0))
    # right: numbers
    add_text(s, Inches(8.4), Inches(2.0), Inches(4.5), Inches(0.5),
             "APₗ (large)", size=15, color=LIGHT)
    add_text(s, Inches(8.4), Inches(2.4), Inches(4.5), Inches(1.4),
             "67.7", size=84, bold=True, color=GREEN)
    add_text(s, Inches(8.4), Inches(4.0), Inches(4.5), Inches(0.5),
             "APₛ (small)", size=15, color=LIGHT)
    add_text(s, Inches(8.4), Inches(4.4), Inches(4.5), Inches(1.4),
             "34.7", size=84, bold=True, color=RED)
    add_text(s, Inches(8.4), Inches(6.0), Inches(4.5), Inches(0.6),
             "33-point gap. Why?",
             size=18, color=DARK, italic=True)
    add_footer(s, 2, TOTAL)

    # ---------- 3. Baseline model ----------
    s = add_blank_slide(prs)
    add_title(s, "Baseline RT-DETR")
    add_image(s, BUILD / "diag_baseline.png",
              Inches(0.8), Inches(2.5), w=Inches(11.7))
    add_text(s, Inches(0.7), Inches(5.5), Inches(12.0), Inches(0.6),
             "Encoder memory is computed once. The decoder cannot revise it.",
             size=22, color=DARK, align=PP_ALIGN.CENTER)
    add_footer(s, 3, TOTAL)

    # ---------- 4. Our idea ----------
    s = add_blank_slide(prs)
    add_title(s, "Our idea: let the model refine its own features")
    add_image(s, BUILD / "diag_idea.png",
              Inches(0.8), Inches(2.2), w=Inches(11.7))
    add_text(s, Inches(0.7), Inches(5.7), Inches(12.0), Inches(0.6),
             "Feed early decoder predictions back into the encoder memory.",
             size=22, color=DARK, align=PP_ALIGN.CENTER)
    add_footer(s, 4, TOTAL)

    # ---------- 5. MATH 1: Attention ----------
    s = add_blank_slide(prs)
    add_title(s, "Attention, in one line")
    add_image(s, BUILD / "math_attention.png",
              Inches(0.5), Inches(1.5), w=Inches(12.3))
    add_footer(s, 5, TOTAL)

    # ---------- 6. MATH 2: Feedback formula ----------
    s = add_blank_slide(prs)
    add_title(s, "Our feedback rule")
    add_image(s, BUILD / "math_feedback.png",
              Inches(0.5), Inches(1.5), w=Inches(12.3))
    add_footer(s, 6, TOTAL)

    # ---------- 7. v1 first attempt ----------
    s = add_blank_slide(prs)
    add_title(s, "First attempt (v1)")
    add_text(s, Inches(0.7), Inches(2.0), Inches(12.0), Inches(0.6),
             "Add the feedback module. Train. Toggle it off at inference.",
             size=22, color=DARK, align=PP_ALIGN.CENTER, italic=True)
    add_text(s, Inches(0.7), Inches(3.6), Inches(12.0), Inches(2.0),
             "Δ APₛ  =  0.00",
             size=110, bold=True, color=RED, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(5.9), Inches(12.0), Inches(0.5),
             "The mechanism contributed nothing.",
             size=20, color=DARK, align=PP_ALIGN.CENTER)
    add_footer(s, 7, TOTAL)

    # ---------- 8. Why v1 failed ----------
    s = add_blank_slide(prs)
    add_title(s, "Why v1 failed: the gate collapsed")
    add_image(s, FIG_DIR / "gate_reparam.png",
              Inches(4.5), Inches(1.6), w=Inches(8.5))
    add_text(s, Inches(0.6), Inches(2.4), Inches(3.8), Inches(0.5),
             "Plain sigmoid gate.",
             size=22, bold=True, color=DARK)
    add_text(s, Inches(0.6), Inches(3.0), Inches(3.8), Inches(2.5),
             "α drifts to −∞ → gate ≈ 0\n\nFeedback signal is multiplied by zero before reaching memory.",
             size=15, color=DARK)
    add_footer(s, 8, TOTAL)

    # ---------- 9. Our fix (v2) ----------
    s = add_blank_slide(prs)
    add_title(s, "Two fixes — zero new parameters")
    add_image(s, BUILD / "diag_v2_fixes.png",
              Inches(1.0), Inches(2.2), w=Inches(11.3))
    add_text(s, Inches(0.7), Inches(6.1), Inches(12.0), Inches(0.5),
             "Reparameterize the gate. Refine only the levels small objects live in.",
             size=18, color=DARK, italic=True, align=PP_ALIGN.CENTER)
    add_footer(s, 9, TOTAL)

    # ---------- 10. Training results ----------
    s = add_blank_slide(prs)
    add_title(s, "Training: v2 climbs faster, ends higher")
    add_image(s, FIG_DIR / "learning_curves.png",
              Inches(2.4), Inches(1.6), w=Inches(8.5))
    add_text(s, Inches(0.7), Inches(6.4), Inches(12.0), Inches(0.4),
             "v2 first crosses v1's final 33.91 at epoch 8 — four epochs early.",
             size=14, color=LIGHT, italic=True, align=PP_ALIGN.CENTER)
    add_footer(s, 10, TOTAL)

    # ---------- 11. Final performance ----------
    s = add_blank_slide(prs)
    add_title(s, "Final performance (with feedback ON)")
    # left card
    c1 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(1.0), Inches(2.2), Inches(5.3), Inches(3.5))
    c1.fill.solid(); c1.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF7)
    c1.line.color.rgb = LIGHT; c1.line.width = Pt(0.75)
    c1.shadow.inherit = False
    add_text(s, Inches(1.0), Inches(2.4), Inches(5.3), Inches(0.5),
             "v1", size=24, color=LIGHT, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.0), Inches(2.95), Inches(5.3), Inches(2.0),
             "33.91", size=88, bold=True, color=LIGHT, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.0), Inches(5.0), Inches(5.3), Inches(0.5),
             "APₛ",
             size=18, color=LIGHT, align=PP_ALIGN.CENTER, italic=True)
    # right card
    c2 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(7.0), Inches(2.2), Inches(5.3), Inches(3.5))
    c2.fill.solid(); c2.fill.fore_color.rgb = RGBColor(0xE8, 0xF7, 0xE8)
    c2.line.color.rgb = GREEN; c2.line.width = Pt(2)
    c2.shadow.inherit = False
    add_text(s, Inches(7.0), Inches(2.4), Inches(5.3), Inches(0.5),
             "v2", size=24, color=GREEN, align=PP_ALIGN.CENTER, bold=True)
    add_text(s, Inches(7.0), Inches(2.95), Inches(5.3), Inches(2.0),
             "34.30", size=88, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    add_text(s, Inches(7.0), Inches(5.0), Inches(5.3), Inches(0.5),
             "APₛ    +0.4",
             size=18, color=GREEN, align=PP_ALIGN.CENTER, italic=True)
    add_footer(s, 11, TOTAL)

    # ---------- 12. Ablation explained ----------
    s = add_blank_slide(prs)
    add_title(s, "Ablation: same model, only the switch changes")
    add_image(s, BUILD / "diag_ablation.png",
              Inches(1.0), Inches(2.5), w=Inches(11.3))
    add_text(s, Inches(0.7), Inches(6.0), Inches(12.0), Inches(0.5),
             "Toggle feedback ON or OFF at inference. Everything else identical.",
             size=18, color=DARK, italic=True, align=PP_ALIGN.CENTER)
    add_footer(s, 12, TOTAL)

    # ---------- 13. Causal result (BIG) ----------
    s = add_blank_slide(prs)
    add_title(s, "Causal effect of feedback")
    # 3 columns: ON, OFF, Δ
    cols = [
        ("ON",  f"{APS_ON:.2f}",  GREEN, "feedback enabled"),
        ("OFF", f"{APS_OFF:.2f}", LIGHT, "feedback bypassed"),
        ("Δ",   f"+{DELTA:.2f}",  NAVY,  "the contribution"),
    ]
    for i, (label, val, c, sub) in enumerate(cols):
        x = Inches(0.7 + i * 4.15)
        add_text(s, x, Inches(2.0), Inches(4.0), Inches(0.6),
                 label, size=24, color=LIGHT, align=PP_ALIGN.CENTER)
        add_text(s, x, Inches(2.7), Inches(4.0), Inches(2.2),
                 val, size=110, bold=True, color=c, align=PP_ALIGN.CENTER)
        add_text(s, x, Inches(5.2), Inches(4.0), Inches(0.5),
                 sub, size=14, color=LIGHT, italic=True, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(6.2), Inches(12.0), Inches(0.6),
             "Feedback has a real causal impact.",
             size=22, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_footer(s, 13, TOTAL)

    # ---------- 14. Resolution effect ----------
    s = add_blank_slide(prs)
    add_title(s, "Higher resolution → bigger gain")
    # left
    add_text(s, Inches(0.7), Inches(2.4), Inches(5.8), Inches(0.5),
             "640 × 640", size=22, color=LIGHT, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(3.0), Inches(5.8), Inches(2.0),
             f"{APS_ON:.2f}", size=88, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    # arrow
    arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                              Inches(5.8), Inches(3.6), Inches(1.7), Inches(0.6))
    arr.fill.solid(); arr.fill.fore_color.rgb = NAVY
    arr.line.fill.background()
    # right
    add_text(s, Inches(6.8), Inches(2.4), Inches(5.8), Inches(0.5),
             "800 × 800", size=22, color=NAVY, align=PP_ALIGN.CENTER, bold=True)
    add_text(s, Inches(6.8), Inches(3.0), Inches(5.8), Inches(2.0),
             f"{APS_800:.2f}", size=88, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(5.6), Inches(12.0), Inches(0.5),
             f"+{APS_800 - APS_ON:.2f} from more pixels.",
             size=22, color=DARK, italic=True, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(6.2), Inches(12.0), Inches(0.5),
             "Feedback helps. Spatial sampling helps too.",
             size=16, color=LIGHT, align=PP_ALIGN.CENTER)
    add_footer(s, 14, TOTAL)

    # ---------- 15. Trade-offs ----------
    s = add_blank_slide(prs)
    add_title(s, "Trade-offs")
    # green +
    plus = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               Inches(1.0), Inches(2.5), Inches(5.3), Inches(3.5))
    plus.fill.solid(); plus.fill.fore_color.rgb = RGBColor(0xE8, 0xF7, 0xE8)
    plus.line.color.rgb = GREEN; plus.line.width = Pt(1.5)
    plus.shadow.inherit = False
    add_text(s, Inches(1.0), Inches(2.7), Inches(5.3), Inches(1.0),
             "+", size=88, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.0), Inches(4.3), Inches(5.3), Inches(0.6),
             "Better small-object detection",
             size=22, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.0), Inches(5.1), Inches(5.3), Inches(0.5),
             "+0.99 APₛ causal contribution",
             size=14, color=LIGHT, italic=True, align=PP_ALIGN.CENTER)
    # red −
    minus = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(7.0), Inches(2.5), Inches(5.3), Inches(3.5))
    minus.fill.solid(); minus.fill.fore_color.rgb = RGBColor(0xFD, 0xEC, 0xEC)
    minus.line.color.rgb = RED; minus.line.width = Pt(1.5)
    minus.shadow.inherit = False
    add_text(s, Inches(7.0), Inches(2.7), Inches(5.3), Inches(1.0),
             "−", size=88, bold=True, color=RED, align=PP_ALIGN.CENTER)
    add_text(s, Inches(7.0), Inches(4.3), Inches(5.3), Inches(0.6),
             "Slower inference (FPS drops)",
             size=22, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    add_text(s, Inches(7.0), Inches(5.1), Inches(5.3), Inches(0.5),
             "53 FPS → 26 FPS  (cost is the P2 level, not feedback)",
             size=14, color=LIGHT, italic=True, align=PP_ALIGN.CENTER)
    add_footer(s, 15, TOTAL)

    # ---------- 16. Conclusion ----------
    s = add_blank_slide(prs)
    add_title(s, "Conclusion")
    bullets = [
        ("Feedback improves small-object detection",
         "+0.99 APₛ causal contribution at inference."),
        ("Only works with proper design",
         "Floor the gate; restrict to small-object levels."),
        ("Modest but real",
         "Consistent across metrics; reproducible from the same checkpoint."),
    ]
    for i, (head, sub) in enumerate(bullets):
        y = Inches(2.0 + i * 1.55)
        # navy circle
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                   Inches(0.9), y + Inches(0.15),
                                   Inches(0.55), Inches(0.55))
        circ.fill.solid(); circ.fill.fore_color.rgb = NAVY
        circ.line.fill.background()
        add_text(s, Inches(0.9), y + Inches(0.18), Inches(0.55), Inches(0.55),
                 str(i + 1), size=20, bold=True, color=BG, align=PP_ALIGN.CENTER)
        add_text(s, Inches(1.7), y + Inches(0.05), Inches(11.0), Inches(0.6),
                 head, size=24, bold=True, color=DARK)
        add_text(s, Inches(1.7), y + Inches(0.7), Inches(11.0), Inches(0.6),
                 sub, size=15, color=LIGHT, italic=True)
    add_footer(s, 16, TOTAL)

    # ---------- 17. Limitations ----------
    s = add_blank_slide(prs)
    add_title(s, "Limitations")
    items = [
        ("Single training run", "no multi-seed variance"),
        ("Below paper baseline",  "13-epoch finetune vs 72-epoch from scratch"),
        ("Increased latency",  "~2× slower (cost is the P2 level)"),
    ]
    for i, (head, sub) in enumerate(items):
        y = Inches(2.0 + i * 1.55)
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                   Inches(0.9), y + Inches(0.18),
                                   Inches(0.5), Inches(0.5))
        circ.fill.solid(); circ.fill.fore_color.rgb = LIGHT
        circ.line.fill.background()
        add_text(s, Inches(0.9), y + Inches(0.21), Inches(0.5), Inches(0.5),
                 "—", size=20, bold=True, color=BG, align=PP_ALIGN.CENTER)
        add_text(s, Inches(1.7), y + Inches(0.05), Inches(11.0), Inches(0.6),
                 head, size=24, bold=True, color=DARK)
        add_text(s, Inches(1.7), y + Inches(0.7), Inches(11.0), Inches(0.6),
                 sub, size=15, color=LIGHT, italic=True)
    add_footer(s, 17, TOTAL)

    # ---------- 18. Thank you ----------
    s = add_blank_slide(prs)
    add_text(s, Inches(0.7), Inches(2.4), Inches(12.0), Inches(1.5),
             "Thank you.",
             size=80, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(4.2), Inches(12.0), Inches(0.8),
             "Questions?",
             size=40, color=NAVY, align=PP_ALIGN.CENTER)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                              Inches(6.17), Inches(5.4), Inches(1.0), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    add_text(s, Inches(0.7), Inches(5.7), Inches(12.0), Inches(0.4),
             "Hatem Saadallah  ·  Nour Jennane",
             size=16, color=DARK, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.7), Inches(6.2), Inches(12.0), Inches(0.4),
             "github.com/HatemSaadallah/feedback-augmented-rtdetr",
             size=12, color=LIGHT, align=PP_ALIGN.CENTER)

    prs.save(OUT)
    print(f"\nSaved: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {TOTAL} slides)")


if __name__ == "__main__":
    render_all_figures()
    build()
