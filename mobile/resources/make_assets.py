"""Generate the CrackScan launcher icon and splash screen.

    python mobile/resources/make_assets.py

Then (from mobile/):
    npx @capacitor/assets generate --android

Output: resources/icon.png (1024x1024) and resources/splash.png (2732x2732).
"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE_TOP = (47, 123, 255)
BLUE_BOTTOM = (16, 74, 200)
BG = (238, 244, 251)


def rounded_mask(size, radius):
    mask = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def gradient(size, top, bottom):
    img = Image.new('RGB', (1, size))
    px = img.load()
    for y in range(size):
        t = y / max(1, size - 1)
        px[0, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return img.resize((size, size))


def crack_points(size):
    """The lightning-like crack from the app's mark, scaled to `size`."""
    pts = [(0.50, 0.13), (0.575, 0.385), (0.425, 0.50), (0.585, 0.655),
           (0.475, 0.875)]
    return [(x * size, y * size) for x, y in pts]


def draw_crack(draw, size, color=(255, 255, 255), width=None):
    width = width or int(size * 0.072)
    pts = crack_points(size)
    draw.line(pts, fill=color, width=width, joint='curve')
    r = width / 2.0
    for x, y in pts:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def make_icon(size=1024):
    base = gradient(size, BLUE_TOP, BLUE_BOTTOM)
    glow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((size * 0.12, size * 0.02, size * 0.88, size * 0.62),
               fill=(255, 255, 255, 26))
    base = Image.alpha_composite(base.convert('RGBA'), glow)

    d = ImageDraw.Draw(base)
    draw_crack(d, size)

    base.putalpha(rounded_mask(size, int(size * 0.235)))
    out = os.path.join(HERE, 'icon.png')
    base.save(out)
    print('wrote', out)

    # Android adaptive icon foreground (crack on transparent, safe 66% area)
    fg = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw_crack(ImageDraw.Draw(fg), size)
    # scale down into the safe zone
    small = fg.resize((int(size * 0.62), int(size * 0.62)), Image.LANCZOS)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    off = (size - small.size[0]) // 2
    canvas.alpha_composite(small, (off, off))
    fg_out = os.path.join(HERE, 'icon-foreground.png')
    canvas.save(fg_out)

    bg = Image.new('RGBA', (size, size), BLUE_BOTTOM + (255,))
    bg_out = os.path.join(HERE, 'icon-background.png')
    bg.save(bg_out)
    print('wrote', fg_out, 'and', bg_out)


def make_splash(size=2732):
    img = Image.new('RGBA', (size, size), BG + (255,))
    logo_size = int(size * 0.26)
    icon = Image.open(os.path.join(HERE, 'icon.png')).convert('RGBA')
    icon = icon.resize((logo_size, logo_size), Image.LANCZOS)
    pos = ((size - logo_size) // 2, int(size * 0.36))
    img.alpha_composite(icon, pos)

    d = ImageDraw.Draw(img)
    text = 'CrackScan'
    # PIL's default font is tiny; draw the wordmark as simple bars if no TTF
    try:
        from PIL import ImageFont
        font = ImageFont.truetype('segoeuib.ttf', int(size * 0.052))
    except Exception:
        font = None
    if font:
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        d.text(((size - tw) / 2, pos[1] + logo_size + size * 0.03), text,
               font=font, fill=(22, 42, 66, 255))

    out = os.path.join(HERE, 'splash.png')
    img.convert('RGB').save(out)
    print('wrote', out)


if __name__ == '__main__':
    make_icon()
    make_splash()
