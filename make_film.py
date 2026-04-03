"""
אדונית השוקולד — Cinematic Pralines Film v2
Source: photos/הpic for video לסרטון.png
Target: ~22 seconds, 1080×1080, H.264 high quality
"""

import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from bidi.algorithm import get_display
import imageio.v2 as iio

# ── PATHS ────────────────────────────────────────────────────────────────
SRC = "photos/הpic for video לסרטון.png"
OUT = "pralines-film.mp4"

# ── SETTINGS ─────────────────────────────────────────────────────────────
W, H = 1080, 1080
FPS  = 30

# ── FONTS ────────────────────────────────────────────────────────────────
F_HEBREW = "/System/Library/Fonts/ArialHB.ttc"          # Hebrew glyphs
F_SERIF  = "/System/Library/Fonts/Supplemental/Didot.ttc"  # Latin serif
F_THIN   = "/System/Library/Fonts/Supplemental/Futura.ttc" # Latin thin

def font(path, size):
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

# ── COLOURS ──────────────────────────────────────────────────────────────
CHOC      = (42,  26,  20)
CREAM     = (245, 237, 224)
IVORY     = (250, 246, 238)
GOLD      = (184, 148, 90)
WARM_GRAY = (196, 176, 154)
BLACK     = (13,   9,   6)

def rgba(c, a): return (*c, int(max(0,min(1,a))*255))

# ── EASING ───────────────────────────────────────────────────────────────
def eio(t):
    t = max(0.0, min(1.0, t))
    return 4*t**3 if t < 0.5 else 1 - (-2*t+2)**3/2

def eout(t):
    t = max(0.0, min(1.0, t))
    return 1.0 if t==1 else 1 - 2**(-10*t)

def ein(t):
    t = max(0.0, min(1.0, t))
    return 0.0 if t==0 else 2**(10*t-10)

# ── PRALINE DATA ─────────────────────────────────────────────────────────
# cx, cy = centre of praline as fraction of source image dimensions
# cr     = half-size of crop square (fraction of source WIDTH)
#          Increased to ~2× to keep quality high (less extreme zoom)
PRALINES = [
    dict(num="01", he="ירח שוקולד",  en="Milk Chocolate · Fleur de Sel",
         cx=0.085, cy=0.500, cr=0.130),
    dict(num="02", he="כדור טרופל",  en="Dark Ganache · Sea Salt",
         cx=0.215, cy=0.494, cr=0.120),
    dict(num="03", he="חמסה לבנה",   en="White Chocolate · Rose Water",
         cx=0.335, cy=0.494, cr=0.115),
    dict(num="04", he="טיפת פנינה",  en="Pure White · Vanilla Bean",
         cx=0.445, cy=0.490, cr=0.110),
    dict(num="05", he="לב שוקולד",   en="Milk Chocolate · Praline Cream",
         cx=0.548, cy=0.496, cr=0.120),
    dict(num="06", he="עלי ורד",     en="Dark Chocolate · Dried Rose Petals",
         cx=0.648, cy=0.490, cr=0.125),
    dict(num="07", he="לוח אגוזים",  en="Milk Chocolate Bark · Roasted Hazelnuts",
         cx=0.808, cy=0.502, cr=0.160),
]

# ── LOAD SOURCE ──────────────────────────────────────────────────────────
print("Loading source image…")
SRC_IMG = Image.open(SRC).convert("RGB")
IW, IH  = SRC_IMG.size
print(f"  Source: {IW}×{IH}")

# ── CROP HELPERS ─────────────────────────────────────────────────────────
def full_crop():
    """Centre-square crop of the full source."""
    side = min(IW, IH)
    x0 = (IW-side)//2;  y0 = (IH-side)//2
    return (x0, y0, x0+side, y0+side)

def praline_crop(p):
    """Tight crop centred on a praline."""
    cx = int(p["cx"]*IW);  cy = int(p["cy"]*IH)
    r  = int(p["cr"]*IW)
    x0, y0 = max(0,cx-r), max(0,cy-r)
    x1, y1 = min(IW,cx+r), min(IH,cy+r)
    side = min(x1-x0, y1-y0)
    mx, my = (x0+x1)//2, (y0+y1)//2
    h = side//2
    return (mx-h, my-h, mx+h, my+h)

def interp_crop(c0, c1, t):
    return tuple(int(c0[i]+(c1[i]-c0[i])*t) for i in range(4))

def render(crop, sharpen=False):
    box = (max(0,crop[0]), max(0,crop[1]),
           min(IW,crop[2]), min(IH,crop[3]))
    region = SRC_IMG.crop(box)
    frame  = region.resize((W,H), Image.LANCZOS)
    if sharpen:
        frame = frame.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=2))
    return frame

FULL_CROP = full_crop()
PCROP     = [praline_crop(p) for p in PRALINES]

# ── VIGNETTE (precomputed) ───────────────────────────────────────────────
print("Building vignette…")
_vy, _vx = np.mgrid[0:H, 0:W]
_dx = (_vx - W/2) / (W*0.5)
_dy = (_vy - H/2) / (H*0.5)
_d  = np.sqrt(_dx**2 + _dy**2)
VIG = np.clip(_d**1.9 * 0.72, 0, 1).astype(np.float32)

def apply_vignette(frame, strength=1.0):
    if strength <= 0: return frame
    arr  = np.array(frame, dtype=np.float32)
    mask = 1.0 - VIG * strength
    arr  = arr * mask[:,:,None]
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

def darken(frame, alpha):
    if alpha <= 0: return frame
    if isinstance(frame, np.ndarray):
        arr = frame.astype(np.float32)
    else:
        arr = np.array(frame, dtype=np.float32)
    arr = arr * (1-alpha) + np.array(BLACK, dtype=np.float32) * alpha
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

# ── TEXT HELPERS ─────────────────────────────────────────────────────────
def draw_praline_label(frame, p, a_num, a_he, a_en, a_line, line_w):
    """Bottom-left cinematic label."""
    ov = Image.new("RGBA", (W,H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)

    pad  = int(W*0.075)
    base = H - int(H*0.072)

    # gradient scrim
    scrim = Image.new("RGBA", (W, int(H*0.44)), (0,0,0,0))
    sd    = ImageDraw.Draw(scrim)
    rows  = int(H*0.44)
    for gy in range(rows):
        av = int(210 * (gy/rows)**0.50)
        sd.line([(0,gy),(W,gy)], fill=(13,9,6,av))
    ov.paste(scrim, (0, H-rows), scrim)

    # gold rule
    if line_w > 0 and a_line > 0:
        ly = base - int(H*0.200)
        d.rectangle([pad, ly, pad+line_w, ly+1],
                    fill=rgba(GOLD, 0.75*a_line))

    # number
    fn = font(F_THIN, int(W*0.020))
    d.text((pad, base-int(H*0.178)), p["num"],
           font=fn, fill=rgba(GOLD, a_num))

    # Hebrew name  — RTL via bidi
    fh   = font(F_HEBREW, int(W*0.072))
    name = get_display(p["he"])
    d.text((pad, base-int(H*0.148)), name,
           font=fh, fill=rgba(IVORY, a_he))

    # English descriptor
    fe  = font(F_THIN, int(W*0.020))
    d.text((pad, base-int(H*0.050)), p["en"].upper(),
           font=fe, fill=rgba(WARM_GRAY, a_en))

    out = Image.alpha_composite(frame.convert("RGBA"), ov)
    return out.convert("RGB")

def draw_progress_dots(frame, active):
    """Right-side vertical dot navigation."""
    ov = Image.new("RGBA", (W,H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)
    n   = len(PRALINES)
    dr  = int(W*0.006)
    gap = int(W*0.032)
    tot = n*(dr*2) + (n-1)*gap
    sx  = W - int(W*0.052)
    sy  = H//2 - tot//2
    for i in range(n):
        cy_ = sy + i*(dr*2+gap) + dr
        if i == active:
            d.ellipse([sx-dr, cy_-dr, sx+dr, cy_+dr],
                      fill=rgba(GOLD, 0.90))
        else:
            d.ellipse([sx-dr+1, cy_-dr+1, sx+dr-1, cy_+dr-1],
                      outline=rgba(WARM_GRAY, 0.35), width=1)
    out = Image.alpha_composite(frame.convert("RGBA"), ov)
    return out.convert("RGB")

def draw_opening_title(frame, a_brand, a_sub, line_w):
    """Centred cinematic title card. Hebrew rendered with Hebrew font."""
    ov = Image.new("RGBA", (W,H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)

    # soft dark ellipse scrim
    scrimr = Image.new("RGBA", (W,H), (13,9,6,110))
    mask   = Image.new("L", (W,H), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([W//5, H//5, 4*W//5, 4*H//5], fill=180)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=W//4))
    ov.paste(scrimr, mask=mask)

    cy = H//2

    # Brand name in Hebrew → use Hebrew font
    f_brand_he = font(F_HEBREW, int(W*0.080))
    brand_t    = get_display("אדונית השוקולד")
    bb = f_brand_he.getbbox(brand_t)
    bw = bb[2]-bb[0]
    bx = (W-bw)//2
    d.text((bx, cy - int(H*0.085)), brand_t,
           font=f_brand_he, fill=rgba(IVORY, a_brand))

    # gold rule
    if line_w > 0:
        lx = W//2 - line_w//2
        d.rectangle([lx, cy+int(H*0.005), lx+line_w, cy+int(H*0.005)+1],
                    fill=rgba(GOLD, 0.80*a_sub))

    # sub (Latin)
    f_sub = font(F_THIN, int(W*0.022))
    sub_t = "HANDCRAFTED CHOCOLATE BOUTIQUE"
    bb2   = f_sub.getbbox(sub_t)
    sw    = bb2[2]-bb2[0]
    d.text(((W-sw)//2, cy+int(H*0.038)), sub_t,
           font=f_sub, fill=rgba(GOLD, a_sub))

    out = Image.alpha_composite(frame.convert("RGBA"), ov)
    return out.convert("RGB")

def outro_card(a_line, line_h, a_name, a_sub):
    f  = Image.new("RGB", (W,H), CHOC)
    ov = Image.new("RGBA", (W,H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)
    # warm gradient
    for y in range(H):
        t = y/H
        r = int(CHOC[0]*(1-t) + 58*t)
        g = int(CHOC[1]*(1-t) + 34*t)
        b = int(CHOC[2]*(1-t) + 24*t)
        d.line([(0,y),(W,y)], fill=(r,g,b,255))
    # vertical gold line
    if line_h > 0:
        lx = W//2
        ly = H//2 - int(H*0.20)
        d.line([(lx,ly),(lx,ly+line_h)], fill=rgba(GOLD,0.75*a_line), width=1)
    # brand name in Hebrew font
    fn     = font(F_HEBREW, int(W*0.082))
    name_t = get_display("אדונית השוקולד")
    bb     = fn.getbbox(name_t)
    d.text(((W-(bb[2]-bb[0]))//2, H//2-int(H*0.065)), name_t,
           font=fn, fill=rgba(IVORY, a_name))
    # sub
    fs    = font(F_THIN, int(W*0.021))
    sub_t = "HANDCRAFTED CHOCOLATE BOUTIQUE"
    bb2   = fs.getbbox(sub_t)
    d.text(((W-(bb2[2]-bb2[0]))//2, H//2+int(H*0.058)), sub_t,
           font=fs, fill=rgba(GOLD, a_sub))
    base = Image.alpha_composite(f.convert("RGBA"), ov)
    return base.convert("RGB")

# ── FRAME BUFFER ─────────────────────────────────────────────────────────
frames = []
fc = lambda s: max(1, int(s * FPS))

# ════════════════════════════════════════════════════════════════════════
#  SCENE 0 — FULL IMAGE FADE IN + TITLE  (3.2 s)
# ════════════════════════════════════════════════════════════════════════
print("Scene 0: Opening title…")

# fade in from black (0.9 s)
for i in range(fc(0.9)):
    t = eout(i/fc(0.9))
    f = render(FULL_CROP)
    f = darken(f, 1-t)
    frames.append(np.array(f))

# title reveal + hold (1.8 s)
for i in range(fc(1.8)):
    t  = i/fc(1.8)
    f  = render(FULL_CROP)
    f  = darken(f, 0.28)
    ab = eout(min(1, t*2.5))
    as_ = eout(max(0,(t-0.25)*2.8))
    lw = int(eout(max(0,(t-0.15)*3.5))*int(W*0.11))
    f  = draw_opening_title(f, ab, as_, lw)
    frames.append(np.array(f))

# title fade out (0.6 s)
for i in range(fc(0.6)):
    t  = i/fc(0.6)
    f  = render(FULL_CROP)
    a  = 1 - ein(t)
    f  = darken(f, 0.28*(1-ein(t)) + ein(t))
    f  = draw_opening_title(f, a, a, int(a*W*0.11))
    frames.append(np.array(f))

# ════════════════════════════════════════════════════════════════════════
#  ROW INTRO — show all pralines (0.9 s)
# ════════════════════════════════════════════════════════════════════════
print("Row intro…")
for i in range(fc(0.9)):
    t = i/fc(0.9)
    f = render(FULL_CROP)
    # gentle fade in from black after title
    fa = eout(t*3)
    f = darken(f, max(0, 1-fa))
    f = apply_vignette(f, 0.55)
    frames.append(np.array(f))

# ════════════════════════════════════════════════════════════════════════
#  SCENES 1–7  — per praline: ZOOM IN → HOLD → ZOOM OUT → ROW
#
#  For each praline:
#    1. ZOOM IN  (0.55 s): full row → praline close-up
#       Others naturally disappear as crop tightens
#    2. HOLD     (1.0 s):  praline alone on screen with labels
#    3. ZOOM OUT (0.55 s): praline → full row
#       Others naturally re-appear
#    4. ROW PAUSE (0.25 s): full row visible (skipped after last)
# ════════════════════════════════════════════════════════════════════════

DUR_ZOOM_IN  = 0.55
DUR_HOLD     = 1.00
DUR_ZOOM_OUT = 0.55
DUR_ROW_PAUSE= 0.25

for idx, p in enumerate(PRALINES):
    print(f"Praline {idx+1}/7: {p['he']}…")
    tc = PCROP[idx]

    # ── 1. ZOOM IN: FULL_CROP → PCROP[idx] ──────────────────────────────
    # As we zoom in, other pralines naturally slide off screen.
    for i in range(fc(DUR_ZOOM_IN)):
        t    = eio(i/fc(DUR_ZOOM_IN))
        crop = interp_crop(FULL_CROP, tc, t)
        f    = render(crop, sharpen=(t > 0.5))
        # vignette builds up as we zoom in
        vig_str = 0.55 + 0.45 * t
        f = apply_vignette(f, vig_str)
        frames.append(np.array(f))

    # ── 2. HOLD: praline alone, text stagger ────────────────────────────
    for i in range(fc(DUR_HOLD)):
        t  = i/fc(DUR_HOLD)

        # subtle Ken Burns push during hold (~1.5 %)
        scale = 1 + 0.018 * eio(t)
        cx2   = (tc[0]+tc[2])//2
        cy2   = (tc[1]+tc[3])//2
        half  = int((tc[2]-tc[0])//2 / scale)
        nc    = (cx2-half, cy2-half, cx2+half, cy2+half)

        f = render(nc, sharpen=True)
        f = apply_vignette(f, 1.0)

        # text stagger
        an  = eout(min(1, t*3.0))
        ahe = eout(max(0,(t-0.12)*2.5))
        aen = eout(max(0,(t-0.28)*2.8))
        al  = eout(min(1, t*4.0))
        lw  = int(al * W * 0.065)

        f = draw_praline_label(f, p, an, ahe, aen, al, lw)
        f = draw_progress_dots(f, idx)
        frames.append(np.array(f))

    # ── 3. ZOOM OUT: PCROP[idx] → FULL_CROP ─────────────────────────────
    # Others re-appear as the field of view widens.
    for i in range(fc(DUR_ZOOM_OUT)):
        t    = eio(i/fc(DUR_ZOOM_OUT))
        crop = interp_crop(tc, FULL_CROP, t)
        # keep Ken Burns scale at its max, then release
        scale = 1 + 0.018 * (1 - t)
        if t < 0.9:
            cx2 = (tc[0]+tc[2])//2;  cy2 = (tc[1]+tc[3])//2
            half = int((tc[2]-tc[0])//2 / scale)
            inner = (cx2-half, cy2-half, cx2+half, cy2+half)
            crop = interp_crop(inner, FULL_CROP, t)
        f = render(crop, sharpen=(t < 0.5))
        vig_str = 1.0 - 0.45*t
        f = apply_vignette(f, vig_str)
        # text fades out quickly
        a_text = max(0, 1 - t*3.5)
        if a_text > 0:
            f = draw_praline_label(f, p, a_text, a_text, a_text, a_text,
                                   int(a_text*W*0.065))
            f = draw_progress_dots(f, idx)
        frames.append(np.array(f))

    # ── 4. ROW PAUSE (not after last praline) ───────────────────────────
    if idx < len(PRALINES) - 1:
        for i in range(fc(DUR_ROW_PAUSE)):
            f = render(FULL_CROP)
            f = apply_vignette(f, 0.55)
            frames.append(np.array(f))

# ════════════════════════════════════════════════════════════════════════
#  FINAL ROW HOLD  (0.7 s)
# ════════════════════════════════════════════════════════════════════════
print("Final row…")
for i in range(fc(0.7)):
    f = render(FULL_CROP)
    f = apply_vignette(f, 0.55)
    frames.append(np.array(f))

# ════════════════════════════════════════════════════════════════════════
#  OUTRO  (total ~3.8 s)
# ════════════════════════════════════════════════════════════════════════
print("Outro…")

# fade to black  (0.5 s)
for i in range(fc(0.5)):
    t = i/fc(0.5)
    f = render(FULL_CROP)
    f = darken(f, ein(t))
    frames.append(np.array(f))

# outro card reveal  (1.5 s)
for i in range(fc(1.5)):
    t  = i/fc(1.5)
    al = eout(min(1, t*2.0))
    lh = int(al * H * 0.14)
    an = eout(max(0,(t-0.28)*2.2))
    as_= eout(max(0,(t-0.52)*2.8))
    f  = outro_card(al, lh, an, as_)
    frames.append(np.array(f))

# hold outro  (1.1 s)
for _ in range(fc(1.1)):
    f = outro_card(1, int(H*0.14), 1, 1)
    frames.append(np.array(f))

# fade to black  (0.7 s)
for i in range(fc(0.7)):
    t = i/fc(0.7)
    f = outro_card(1, int(H*0.14), 1, 1)
    f = darken(f, ein(t))
    frames.append(np.array(f))

# ════════════════════════════════════════════════════════════════════════
#  WRITE MP4
# ════════════════════════════════════════════════════════════════════════
total_s = len(frames)/FPS
print(f"\nTotal: {len(frames)} frames  ({total_s:.1f}s)")
print(f"Writing {OUT}…")

writer = iio.get_writer(
    OUT, fps=FPS, codec="libx264",
    quality=10, macro_block_size=None,
    ffmpeg_params=[
        "-crf",    "14",
        "-preset", "slow",
        "-pix_fmt","yuv420p",
        "-movflags","+faststart",
    ],
)
for i, frm in enumerate(frames):
    writer.append_data(frm)
    if i % FPS == 0:
        print(f"  {i//FPS}s / {int(total_s)}s", end="\r")
writer.close()

print(f"\n✓  Saved: {OUT}  ({total_s:.1f}s)")
