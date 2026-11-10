import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

G = "/mnt/d/坚果云存储/复旦事宜/科研/groupmeeting/2026/DDPM_261013.pptx"
P = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#/results/plots/"
prs = Presentation(G.replace(".pptx", "_backup_before_claude.pptx"))
S = list(prs.slides)
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def set_title(slide, text):
    tf = slide.shapes.title.text_frame
    r = tf.paragraphs[0].runs
    r[0].text = text
    for x in r[1:]:
        x._r.getparent().remove(x._r)


def set_paras(tf, texts):
    """Rewrite paragraphs in place, keeping the first paragraph's formatting for any extra ones."""
    paras = tf.paragraphs
    tmpl = copy.deepcopy(paras[0]._p)
    for p in paras[len(texts):]:
        p._p.getparent().remove(p._p)
    for i, t in enumerate(texts):
        if i >= len(tf.paragraphs):
            tf._txBody.append(copy.deepcopy(tmpl))
        p = tf.paragraphs[i]
        runs = p.runs
        if not runs:
            p.add_run()
            runs = p.runs
        runs[0].text = t
        for x in runs[1:]:
            x._r.getparent().remove(x._r)


def replace_picture(slide, old, path, left=None, top=None, width=None, height=None):
    o = shape(slide, old) if isinstance(old, str) else old
    l, t, w, h = (o.left if left is None else left), (o.top if top is None else top), width, height
    if w is None and h is None:
        w, h = o.width, o.height
    pic = slide.shapes.add_picture(path, l, t, w, h)
    o._element.getparent().replace(o._element, pic._element)
    return pic


def textbox(slide, l, t, w, h, lines, size=18, fill=None, font='Calibri', italic=False, bullet=False, align=None):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    if fill:
        tb.fill.solid(); tb.fill.fore_color.rgb = RGBColor.from_string(fill)
        tf.margin_left = tf.margin_right = Inches(0.02)
        tf.margin_top = tf.margin_bottom = Inches(0.0)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align: p.alignment = align
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set('marL', str(Inches(0.3))); pPr.set('indent', str(-Inches(0.3)))
            bu = pPr.makeelement('{%s}buChar' % NS['a'], {'char': '•'})
            pPr.append(bu)
            p.space_after = Pt(8)
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.italic = italic
        if font: r.font.name = font
        r.font.color.rgb = RGBColor(0, 0, 0)
    return tb


# ── 2: credit the borrowed figure
set_paras(shape(S[1], '矩形 9').text_frame,
          ['Ho et al., NeurIPS 2020.  Figure: V. Singh, LearnOpenCV (2023).'])

# ── 3: patch outdated labels in the framework diagram (px/120 = inch on a 1600-px render)
C = dict(align=PP_ALIGN.CENTER, font='Arial')
textbox(S[2], 9.64, 3.43, 1.30, 0.42, ['L1 Loss', 'L = |ε − ε̂|'], size=10, fill='F2D0CB', **C)
textbox(S[2], 4.45, 4.16, 1.55, 0.26, ['β₁ = 1e-4 → βT = 0.5'], size=9, fill='D8E7D5', italic=True, **C)
textbox(S[2], 4.22, 5.87, 2.78, 0.50, ['Reverse step (×50), no random term',
                                      'xₜ₋₁ = (xₜ − βₜ/√(1−ᾱₜ)·ε̂) / √αₜ'], size=10, fill='DCE6FA', **C)
textbox(S[2], 7.80, 5.78, 1.62, 0.56, ['Single deterministic run', '(no multi-shot averaging)'],
        size=10, fill='DED5E5', **C)

# ── 5, 6: titles; 6 also gets the English figure and an explanation
set_title(S[4], 'What happens inside the U-Net')
set_title(S[5], 'What the model does at inference')
replace_picture(S[5], '图片 6', P + 'ddpm_explainer/traj_nonoise_2615keV_en.png',
                left=Inches(5.75), top=Inches(0.12), height=Inches(7.25))
textbox(S[5], 0.35, 1.0, 5.3, 5.8, [
    'Start from Gaussian noise x₅₀ ~ N(0, I); the measurement x̃ enters as a condition at every step',
    'Each step: U-Net predicts ε̂ → clean estimate x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ → update to xₜ₋₁',
    'no_noise sampler: the DDPM update with the random term removed',
    'x̂₀ already looks like a pulse at t = 50, but dividing by √ᾱ₅₀ ≈ 0.006 amplifies small errors (13.5%); by t = 10 the error is 0.08%',
    'Output = x̂₀ at the last step, rescaled by max|x̃|',
], size=16, bullet=True)

# ── 9: training facts
tb9 = shape(S[8], '标题 3')
tb9.top, tb9.height = Inches(0.85), Inches(2.3)
set_paras(tb9.text_frame, [
    '90,500 simulated pulses and synthetic noise windows, re-paired at random every epoch',
    'Split by pulse: 81,450 train / 9,050 validation (90% / 10%)',
    'AdamW, lr 2×10⁻⁴ with cosine decay, batch 16, L1 loss on ε, mixed precision, RTX 3090',
    'Best validation loss at epoch 72 (≈366k steps); that checkpoint is used for all results',
])

# ── 11: evaluation set wording
set_title(S[10], 'Evaluation dataset')
set_paras(shape(S[10], '标题 3').text_frame, [
    'Hold-out pool of 10,000 noise windows, seeds disjoint from the training noise',
    'Each energy point uses one fixed clean template, so the energy spread comes from noise alone',
    'Noise rescaled to a fixed RMS: 1.382 mV (CUORE-like baseline, 2.54 keV FWHM) and 7.5 mV (training noise level, 13.78 keV FWHM)',
    '5,000 events per energy point',
    'Energy points: 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 239, 511, 727, 1461, 2615, 3034 keV',
])
replace_picture(S[10], '图片 4', P + 'slides/validation_dataset_examples.png')

# ── 14: current 15 keV spectra, both noise levels
set_title(S[13], 'Energy resolution: 15 keV example')
replace_picture(S[13], '图片 3', P + 'paper_zeroinit/spectrum_15keV_zeroinit.png',
                left=Inches(0.45), top=Inches(1.25), width=Inches(12.4))

# ── 17: current QA figure (5 % cut) and a definition that matches the code
s17 = S[16]
replace_picture(s17, '图片 4', P + 'efficiency_1p382mV_zeroinit/efficiency_5keV_1p382mV_zeroinit_t5pct_m5pct.png',
                left=Inches(1.4), top=Inches(0.72), width=Inches(10.5))
pic3 = shape(s17, '图片 3'); pic3._element.getparent().remove(pic3._element)
for el in list(s17.shapes._spTree):           # old bullet box (contains OMML math, so drop it whole)
    if 'Tri-exp fit success' in ''.join(el.itertext()):
        s17.shapes._spTree.remove(el)
textbox(s17, 0.5, 5.55, 12.3, 1.4, [
    'Reconstruction efficiency = fraction of events with |E_reco − E_true| / E_true < 5% (calibrated tri-exp fit; failed fits count as failures)',
    'Shown for QA only, not part of the efficiency: fit convergence (χ²/ndf < 10) and shape match, Pearson r(fitted pulse, true template) > 0.90',
], size=16, bullet=True)
textbox(s17, 0.5, 0.75, 1.6, 0.4, ['5 keV, 1.382 mV'], size=12)

# ── 18: zero-init results at 1.382 mV / 7.5 mV
replace_picture(S[17], '图片 6', P + 'paper_zeroinit/reco_efficiency_standalone_zeroinit_1p382mV.png')
replace_picture(S[17], '图片 4', P + 'paper_zeroinit/reco_efficiency_standalone_zeroinit_7p5mV.png')

# ── new summary slide, placed before the backup "What is DDPM?" slide
lay = S[17].slide_layout
ns = prs.slides.add_slide(lay)
ns.shapes.title.text = 'Summary and next steps'
for ph in list(ns.placeholders):
    if ph.placeholder_format.idx != 0 and not ph.has_text_frame or (ph.has_text_frame and ph.placeholder_format.type == 13):
        pass
textbox(ns, 0.6, 0.95, 12.1, 3.0, [
    'A conditional 1D DDPM (15 M parameters) trained on simulated pulses + synthetic CUORE-like noise',
    '15 keV resolution: 0.93 → 0.07 keV FWHM at 1.382 mV, 5.05 → 0.48 keV at 7.5 mV',
    '90% reconstruction-efficiency threshold: 14.3 → 1.7 keV at 1.382 mV, 84.4 → 7.7 keV at 7.5 mV',
    'After denoising the FWHM grows with energy, but stays below the undenoised value at every energy point',
], size=18, bullet=True)
textbox(ns, 0.6, 3.35, 12.1, 0.45, ['Next steps'], size=20)
textbox(ns, 0.6, 3.85, 12.1, 2.3, [
    'Rerun all energy points with the new production sampler (Gaussian start + no_noise); results above start from x_T = 0 (zero-init)',
    'The model was trained around 7.5 mV noise; the 1.382 mV set lies outside the training range',
    'Enlarge the evaluation set for the paper',
], size=18, bullet=True)
# move it before the last original slide
lst = prs.slides._sldIdLst
el = lst[-1]; lst.remove(el); lst.insert(len(lst) - 1, el)

prs.save(G)
print('saved', len(prs.slides), 'slides')
