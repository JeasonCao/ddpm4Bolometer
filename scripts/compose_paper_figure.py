"""
2026-09-22: compose two single-noise-level figures into the side-by-side
(a)/(b) panel the paper uses.

The `figN.<name>.png` files in the notes vault's latex/ folder are horizontal
composites of their `figNa.*` / `figNb.*` siblings. Until now those composites
were assembled by hand, so re-running the plotting pipeline left them stale.
This makes the step reproducible.

Panels are padded to a common height (top-aligned, white background) rather
than rescaled, so the two subfigures keep identical data-area scaling.

Usage
-----
python -u scripts/compose_paper_figure.py left.png right.png --output fig7.png
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

LABEL_H = 46          # strip above the panels for the (a)/(b) labels
GAP = 18              # gap between panels
MARGIN = 10


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def compose(left_path, right_path, output, labels=('(a)', '(b)')):
    ims = [Image.open(p).convert('RGB') for p in (left_path, right_path)]
    h = max(im.height for im in ims)
    w = sum(im.width for im in ims) + GAP + 2 * MARGIN

    canvas = Image.new('RGB', (w, h + LABEL_H + MARGIN), 'white')
    draw = ImageDraw.Draw(canvas)
    font = _font(34)

    x = MARGIN
    for im, lab in zip(ims, labels):
        draw.text((x + 6, MARGIN), lab, fill='black', font=font)
        canvas.paste(im, (x, LABEL_H))
        x += im.width + GAP

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    canvas.save(output)
    print(f'Saved -> {output}  {canvas.size}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('left')
    ap.add_argument('right')
    ap.add_argument('--output', required=True)
    ap.add_argument('--labels', nargs=2, default=['(a)', '(b)'])
    args = ap.parse_args()
    compose(args.left, args.right, args.output, tuple(args.labels))


if __name__ == '__main__':
    main()
