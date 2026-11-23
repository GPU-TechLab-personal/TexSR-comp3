"""Drawing for the montages (variants/eval/montage.py). cv2 + numpy only.

Port of comp1's sheet code - variants/util/report/sheets.py (fit, placeholder,
the GRID / TABLE paddings) and variants/deeptex/render_loss/eval/
deeptex_dt_montage.py (wrapped labels, row-group separators, the methods-down
sheet and the materials-down overview) - plus svbrdf_render_eval's to_cell /
heat_cell. Display conversion is variants/util/report/images.to_display:
16-bit >> 8, no per-cell rescaling, so an all-zero metallic stays black and a
normal shows its stored RGB encoding.

    sheet(title, notes, heads, rows, cell)        methods down, maps / renders across
    overview(title, rows, heads, groups, cell)    materials down, methods across
"""
import cv2
import numpy as np

from variants.util.report.images import to_display

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD = 6
LEFT = 230                       # row-label gutter of the per-material sheet
OV_LEFT = 230                    # row-label gutter of the overview
BG = 18
TEXT = (210, 210, 210)
WHITE = (255, 255, 255)
SEP = (95, 95, 95)               # line between row groups
BOX = (0, 255, 255)              # the zoom window drawn on the full sheet (yellow, BGR)
HEAT_GAIN = 4.0                  # comp1 svbrdf_render_eval.HEAT_GAIN: |pred - GT| x 4
COLORMAP = getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET)
INVALID = (128, 128, 128)        # GT-invalid texels in the heat map
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9]


# --------------------------------------------------------------------- text

def text_w(t, scale):
    return cv2.getTextSize(t, FONT, scale, 1)[0][0]


def fit(text, width, start=0.5, floor=0.3):
    """The largest font scale <= start at which `text` fits in `width` pixels."""
    s = start
    while text_w(text, s) > width and s > floor:
        s -= 0.02
    return s


def wrap(text, width, scale):
    """Greedy word wrap to `width` at `scale`; a single word that still does not
    fit is shrunk when drawn (fit)."""
    lines, cur = [], ""
    for word in text.split(" "):
        t = (cur + " " + word).strip()
        if cur and text_w(t, scale) > width:
            lines.append(cur)
            cur = word
        else:
            cur = t
    return lines + ([cur] if cur else [])


def put_lines(img, lines, x, y, width, scale, colour, step):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x, y + i * step), FONT, fit(t, width, scale), colour, 1,
                    cv2.LINE_AA)


# -------------------------------------------------------------------- cells

def placeholder(size, note=()):
    """A grey "n/a" cell with a few lines saying why."""
    cell = np.full((size, size, 3), 28, np.uint8)
    cv2.putText(cell, "n/a", (size // 2 - 26, size // 2 - 6), FONT, 0.7,
                (110, 110, 110), 2, cv2.LINE_AA)
    for i, line in enumerate(note):
        s = fit(line, size - 8, 0.4)
        w = text_w(line, s)
        cv2.putText(cell, line, (max(4, size // 2 - w // 2), size // 2 + 20 + i * 15),
                    FONT, s, (90, 90, 90), 1, cv2.LINE_AA)
    return cell


def text_cell(size, lines):
    """A black cell with centred grey text (the GT row's heat column)."""
    cell = np.zeros((size, size, 3), np.uint8)
    for i, t in enumerate(lines):
        s = fit(t, size - 16, 0.5)
        w = text_w(t, s)
        cv2.putText(cell, t, (max(4, size // 2 - w // 2), size // 2 - 4 + 22 * i), FONT, s,
                    (150, 150, 150), 1, cv2.LINE_AA)
    return cell


def map_cell(a, size):
    """A stored map array (8/16-bit, grey 2-D or BGR) -> 8-bit BGR cell.
    INTER_AREA when shrinking, INTER_NEAREST when enlarging (to_display), so a
    zoomed cell shows the texels."""
    return to_display(a[..., :3] if a.ndim == 3 else a, size)


def to_u16(img):
    """MatUp's write_image_as_png quantisation (render_core.to_u16)."""
    return np.array(65535.0 * np.clip(img, 0.0, 1.0), dtype=np.uint16)


def render_cell(img, size):
    """Float RGB render in [0,1] -> 8-bit BGR cell, quantised like MatUp's PNG."""
    return to_display(np.ascontiguousarray(to_u16(img)[..., ::-1]), size)


def heat_cell(img, gt, mask_hw, size, label=None):
    """|pred - GT| (mean over RGB) x HEAT_GAIN, inferno; GT-invalid texels grey.
    `label` (e.g. a PSNR) is written in the top-left corner."""
    d = np.abs(img.astype(np.float64) - gt.astype(np.float64)).mean(axis=2) * HEAT_GAIN
    u8 = (np.clip(d, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, COLORMAP)
    bgr[~mask_hw] = INVALID
    interp = cv2.INTER_AREA if bgr.shape[0] > size else cv2.INTER_NEAREST
    bgr = cv2.resize(bgr, (size, size), interpolation=interp)
    if label:
        s = fit(label, size - 12, 0.5)
        cv2.rectangle(bgr, (0, 0), (text_w(label, s) + 10, 22), (0, 0, 0), -1)
        cv2.putText(bgr, label, (5, 16), FONT, s, WHITE, 1, cv2.LINE_AA)
    return bgr


def busiest_window(img, mask_hw, z):
    """(y0, x0) of the z x z window of `img` (float RGB) with the most detail:
    the sum of |grad| of its mean over RGB, GT-invalid texels counted as 0.
    Ties go to the first window in row-major order."""
    h, w = img.shape[:2]
    if z >= min(h, w):
        return 0, 0
    g = img.astype(np.float64).mean(axis=2)
    gy, gx = np.gradient(g)
    e = np.hypot(gx, gy) * mask_hw
    s = cv2.integral(e, sdepth=cv2.CV_64F)
    sums = s[z:, z:] - s[:-z, z:] - s[z:, :-z] + s[:-z, :-z]
    y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(y0), int(x0)


def draw_box(cell, box, side):
    """Outline box = (y0, x0, z) of a side x side image on its (square) cell."""
    y0, x0, z = box
    k = cell.shape[0] / float(side)
    p0 = (int(round(x0 * k)), int(round(y0 * k)))
    p1 = (int(round((x0 + z) * k)) - 1, int(round((y0 + z) * k)) - 1)
    cv2.rectangle(cell, p0, p1, BOX, 2)
    return cell


# ------------------------------------------------------------------- sheets

def separators(groups):
    """Indices i where a separator goes BEFORE row i: the group changes."""
    return [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]


def sheet(title, notes, heads, rows, cell):
    """Methods down, maps and renders across.

    heads  column titles
    rows   [(label, group, [cells])]; each cell is a cell x cell BGR uint8 array
    """
    top = 32 + 18 * len(notes)
    head_lines = [wrap(h, cell - 8, 0.5) for h in heads]
    head = 14 + 19 * max(len(h) for h in head_lines) + 8
    w = LEFT + len(heads) * (cell + PAD) + PAD
    h = top + head + len(rows) * (cell + PAD) + PAD
    out = np.full((h, w, 3), BG, np.uint8)
    cv2.putText(out, title, (10, 22), FONT, fit(title, w - 20, 0.6), WHITE, 1, cv2.LINE_AA)
    for i, t in enumerate(notes):
        cv2.putText(out, t, (10, 22 + 18 * (i + 1)), FONT, fit(t, w - 20, 0.45),
                    (200, 200, 200), 1, cv2.LINE_AA)
    for c, lines in enumerate(head_lines):
        put_lines(out, lines, LEFT + c * (cell + PAD) + 4, top + 20, cell - 8, 0.5, TEXT, 19)
    seps = separators([g for _, g, _ in rows])
    for r, (label, _, cells) in enumerate(rows):
        y = top + head + r * (cell + PAD)
        if r in seps:
            cv2.line(out, (6, y - PAD // 2 - 1), (w - 6, y - PAD // 2 - 1), SEP, 2)
        lines = wrap(label, LEFT - 18, 0.58)
        put_lines(out, lines, 10, y + cell // 2 - (len(lines) - 1) * 23 // 2 + 6,
                  LEFT - 18, 0.58, TEXT, 23)
        for c, cl in enumerate(cells):
            x = LEFT + c * (cell + PAD)
            out[y:y + cell, x:x + cell] = cl
    return out


def overview(title, rows, heads, groups, cell):
    """Materials down, methods across.

    rows    [(material, [cells])], one cell per head (None draws "n/a")
    heads   column titles; groups: one group id per column (separator between groups)
    """
    pad, top = 4, 34
    head_lines = [wrap(t, cell - 6, 0.42) for t in heads]
    head = 12 + 16 * max(len(t) for t in head_lines) + 6
    w = OV_LEFT + len(heads) * (cell + pad) + pad
    h = top + head + len(rows) * (cell + pad) + pad
    out = np.full((h, w, 3), BG, np.uint8)
    cv2.putText(out, title, (10, 24), FONT, fit(title, w - 20, 0.62), WHITE, 1, cv2.LINE_AA)
    seps = separators(groups)
    for c, lines in enumerate(head_lines):
        x = OV_LEFT + c * (cell + pad)
        if c in seps:
            cv2.line(out, (x - pad // 2 - 1, top + 4), (x - pad // 2 - 1, h - 4), SEP, 2)
        put_lines(out, lines, x + 3, top + 18, cell - 6, 0.42, TEXT, 16)
    for r, (name, cells) in enumerate(rows):
        y = top + head + r * (cell + pad)
        cv2.putText(out, name, (8, y + cell // 2), FONT, fit(name, OV_LEFT - 16, 0.46),
                    TEXT, 1, cv2.LINE_AA)
        for c, cl in enumerate(cells):
            x = OV_LEFT + c * (cell + pad)
            out[y:y + cell, x:x + cell] = cl if cl is not None else placeholder(cell)
    return out


def write_png(path, img):
    """Lossless, maximum compression; cv2.imwrite's silent False is fatal."""
    if not cv2.imwrite(path, img, PNG_PARAMS):
        raise SystemExit("cv2.imwrite failed for " + path)
