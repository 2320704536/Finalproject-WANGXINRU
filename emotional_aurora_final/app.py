# app.py  — Fluid Marble Generator (Bright Blue/White)
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO
import math

st.set_page_config(page_title="Fluid Marble — Wang Xinru — Final Project", page_icon="💠", layout="wide")
st.title("💠 Fluid Marble — Wang Xinru — Final Project")

with st.expander("Instructions", expanded=False):
    st.markdown("""
**What this does**
- Generate bright, airy **fluid marble** posters inspired by your references (light blue & white), with silky streamlines and gentle grain.

**How to use**
1. Choose a **Style**: *Flow Lines* (细白线环绕) or *Ribbon Pools* (大片白色负形空间)  
2. Tune **Lines / Thickness / Flow** for形态，**Color**保持蓝白高级感  
3. Toggle **Grain / Bloom / Vignette / Clamp Brightness** 以获得明亮干净的印刷风  
4. Click **Generate** → **Download PNG**
""")

# -------------------------------
# Utilities: fBm noise (value noise upsample)
# -------------------------------
def fbm(h, w, seed=0, octaves=4, base=64, persistence=0.5, lacunarity=2.0):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    scale = base
    for _ in range(octaves):
        gh = max(1, h // max(1, scale))
        gw = max(1, w // max(1, scale))
        g = rng.random((gh, gw)).astype(np.float32)
        # resize to canvas
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        layer = np.asarray(img, dtype=np.float32) / 255.0
        acc += layer * amp
        amp *= persistence
        scale = max(1, int(scale / lacunarity))
    acc -= acc.min()
    m = acc.max()
    if m > 1e-6:
        acc /= m
    return acc

def unit_vec(angle):
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

# -------------------------------
# Flow field (curl-like) & streamline tracer
# -------------------------------
def flow_angles(h, w, seed, scale, swirl, bias):
    n = fbm(h, w, seed=seed, octaves=5, base=max(8, int(scale)), persistence=0.55, lacunarity=2.0)
    # angle field：映射到 [0, 2π)，加一点旋涡感
    theta = (n * 2 * np.pi) + swirl * (np.linspace(-1, 1, w)[None, :] * np.pi * 0.25)
    theta += bias
    return theta.astype(np.float32)

def sample_angle(theta, p):
    h, w = theta.shape
    x = np.clip(p[0], 0, w - 1)
    y = np.clip(p[1], 0, h - 1)
    return theta[int(y), int(x)]

def trace_streamline(theta, p0, step, steps, jitter=0.0):
    pts = [p0.copy()]
    p = p0.copy()
    for _ in range(steps):
        ang = sample_angle(theta, p)
        v = unit_vec(ang)
        if jitter > 0:
            v += (np.random.random(2) - 0.5) * jitter
        p = p + v * step
        pts.append(p.copy())
    return pts

# -------------------------------
# Renderers (two styles)
# -------------------------------
def render_flow_lines(w, h, seed, n_lines, thickness, scale, swirl, bias, line_alpha, bg_rgb, line_rgb, grain):
    theta = flow_angles(h, w, seed, scale, swirl, bias)
    img = Image.new("RGB", (w, h), bg_rgb)
    # 画半透明白线
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rng = np.random.default_rng(seed + 1234)
    steps = int(0.75 * max(w, h) / thickness)

    for _ in range(n_lines):
        # 起点在画面内的随机位置
        p0 = np.array([rng.uniform(0, w), rng.uniform(0, h)], dtype=np.float32)
        pts = trace_streamline(theta, p0, step=thickness*0.9, steps=steps, jitter=0.12)
        # 画折线（抗锯齿：细画+模糊）
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            draw.line((x1, y1, x2, y2), fill=(*line_rgb, int(255 * line_alpha)), width=int(max(1, thickness)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=thickness * 0.35))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    if grain > 0:
        img = add_grain(img, grain)
    return img

def render_ribbon_pools(w, h, seed, bands, scale, swirl, bias, bg_rgb, ribbon_rgb, edge_soft, grain):
    theta = flow_angles(h, w, seed, scale, swirl, bias)
    # 生成多条宽丝带（负形空间形成大白块）
    base = Image.new("RGB", (w, h), bg_rgb)
    band_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(band_img, "L")
    rng = np.random.default_rng(seed + 2222)

    for b in range(bands):
        thickness = int((0.020 + 0.012 * rng.uniform()) * max(w, h))  # 丝带宽
        steps = int(1.2 * max(w, h) / (thickness * 0.6))
        p0 = np.array([rng.uniform(0, w), rng.uniform(0, h / 3)], dtype=np.float32)
        pts = trace_streamline(theta, p0, step=thickness * 0.5, steps=steps, jitter=0.08)
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            draw.line((x1, y1, x2, y2), fill=220, width=thickness)
    # 软边
    band_img = band_img.filter(ImageFilter.GaussianBlur(radius=edge_soft))
    # 将丝带区域抠白
    white = Image.new("RGB", (w, h), ribbon_rgb)
    base = Image.composite(white, base, band_img)
    if grain > 0:
        base = add_grain(base, grain)
    return base

# -------------------------------
# Post: bloom / vignette / clamp brightness / grain
# -------------------------------
def add_grain(img, amount):
    # amount: 0..1 之间，0.15 推荐
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = (np.random.random((h, w, 1)) - 0.5) * (amount * 0.4)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def bloom(img, radius=6, intensity=0.35):
    if intensity <= 0 or radius <= 0:
        return img
    blur = img.filter(ImageFilter.GaussianBlur(radius=radius))
    a = np.array(img, dtype=np.float32)
    b = np.array(blur, dtype=np.float32)
    out = np.clip(a * (1 - intensity) + b * intensity, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def vignette(img, strength=0.15):
    if strength <= 0:
        return img
    w, h = img.size
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(xx ** 2 + yy ** 2)
    mask = 1 - np.clip((r ** 1.7) * strength, 0, 1)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr *= mask[..., None]
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def clamp_brightness(img, target_mean=0.85):
    # 强制整体很亮但不爆（适合你的“必须很亮”要求）
    arr = np.asarray(img, dtype=np.float32) / 255.0
    Y = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    meanY = max(float(Y.mean()), 1e-4)
    gain = target_mean / meanY
    arr = np.clip(arr * gain, 0, 1)
    # 轻柔高光回卷，避免死白
    arr = np.where(arr > 0.98, 0.98 + (arr - 0.98) * 0.3, arr)
    return Image.fromarray((arr * 255).astype(np.uint8))

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("1) Canvas")
W = st.sidebar.number_input("Width", 600, 2400, 1200, 50)
H = st.sidebar.number_input("Height", 800, 3200, 1800, 50)
seed = st.sidebar.number_input("Seed", 0, 10_000_000, 1234, 1)

st.sidebar.header("2) Style")
style = st.sidebar.selectbox("Style", ["Flow Lines", "Ribbon Pools"], index=0)

st.sidebar.header("3) Flow Field")
scale = st.sidebar.slider("Noise Scale (bigger → smoother)", 8, 256, 96, 4)
swirl = st.sidebar.slider("Swirl", 0.0, 2.0, 0.6, 0.05)
bias = st.sidebar.slider("Angle Bias (rad)", -3.14, 3.14, 0.0, 0.01)

st.sidebar.header("4) Colors (Blue / White)")
# 参考图：浅蓝背景 + 纯白线（或白丝带）
bg_r = st.sidebar.slider("Background Blue R", 150, 210, 188, 1)
bg_g = st.sidebar.slider("Background Blue G", 200, 235, 222, 1)
bg_b = st.sidebar.slider("Background Blue B", 225, 255, 242, 1)
bg_rgb = (bg_r, bg_g, bg_b)
line_rgb = (255, 255, 255)  # 固定白色更干净
ribbon_rgb = (255, 255, 255)

st.sidebar.header("5) Stroke / Ribbons")
if style == "Flow Lines":
    n_lines = st.sidebar.slider("Lines", 180, 1200, 550, 10)
    thickness = st.sidebar.slider("Line Thickness", 1, 8, 3, 1)
    line_alpha = st.sidebar.slider("Line Opacity", 0.4, 1.0, 0.95, 0.01)
else:
    bands = st.sidebar.slider("Ribbon Bands", 6, 40, 18, 1)
    edge_soft = st.sidebar.slider("Ribbon Edge Softness", 2, 24, 10, 1)

st.sidebar.header("6) Finishing")
grain = st.sidebar.slider("Paper Grain", 0.0, 0.5, 0.15, 0.01)
do_bloom = st.sidebar.checkbox("Bloom (very subtle)", value=True)
do_vignette = st.sidebar.checkbox("Vignette (light)", value=False)
target_bright = st.sidebar.slider("Clamp Brightness Target", 0.70, 0.95, 0.88, 0.01)

generate = st.sidebar.button("Generate")

# -------------------------------
# Generate
# -------------------------------
if generate:
    if style == "Flow Lines":
        img = render_flow_lines(
            W, H, seed,
            n_lines=n_lines, thickness=thickness,
            scale=scale, swirl=swirl, bias=bias,
            line_alpha=line_alpha, bg_rgb=bg_rgb, line_rgb=line_rgb, grain=grain
        )
    else:
        img = render_ribbon_pools(
            W, H, seed,
            bands=bands, scale=scale, swirl=swirl, bias=bias,
            bg_rgb=bg_rgb, ribbon_rgb=ribbon_rgb,
            edge_soft=edge_soft, grain=grain
        )

    # 明亮锁定 & 轻后期
    img = clamp_brightness(img, target_mean=target_bright)
    if do_bloom:    img = bloom(img, radius=8, intensity=0.22)
    if do_vignette: img = vignette(img, strength=0.10)
    img = clamp_brightness(img, target_mean=target_bright)  # 再次稳定亮度

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    st.image(img, use_column_width=True)
    st.download_button("💾 Download PNG", data=buf, file_name="fluid_marble.png", mime="image/png")
else:
    st.info("在左侧设置好参数后点击 **Generate**。推荐：Style=Ribbon Pools 可得到大面积白色负形空间的效果；Style=Flow Lines 可得到细密白线环绕的效果。")
