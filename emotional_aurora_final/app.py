# ============================================================
# Emotional Crystal — FINAL VERSION (Part 1 / 6)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date
import math
import random

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Crystal — Final", page_icon="❄️", layout="wide")
st.title("❄️ Emotional Crystal — Final")


# ============================================================
# GLOBAL SEED STATE (核心改动)
# ============================================================

# 每次 Random 或 Keyword Fetch 时都会更新
if "global_seed" not in st.session_state:
    st.session_state["global_seed"] = random.randint(0, 999999)

def refresh_global_seed():
    """每次 Random 或 Keyword Fetch 都调用 → 刷新图案结构"""
    st.session_state["global_seed"] = random.randint(0, 999999)


# ============================================================
# RANDOM EMOTION NAME PREFIX 规则：R1, R2, R3...
# ============================================================
RANDOM_PREFIX = "R1"

def make_random_emotion_name(i):
    return f"{RANDOM_PREFIX}_{i}"


# ============================================================
# Instructions Section
# ============================================================
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

This project transforms emotion data into **cinematic ice-crystal generative visuals**.

**1. Fetch or Generate Data**  
- Enter a keyword → fetch news  
- Or click **Random Generate** → full random crystal mode  

**2. Emotion Filtering**  
- Keyword mode only  
- Random mode ignores Selected Emotions  

**3. Full Crystal Controls**  
- Layers, wobble, seed, softness, alpha, color grading, bloom, vignette...  

**4. CSV Palette**  
- Supports override (Random mode colors still created but can be overwritten)  

**5. Export Image**  
- Download final cinematic crystal artwork as PNG  
""")
# ============================================================
# Emotional Crystal — FINAL VERSION (Part 2 / 6)
# ============================================================

# =========================
# VADER Sentiment Analyzer
# =========================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


# =========================
# NewsAPI - Fetch News
# =========================
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()

        if data.get("status") != "ok" or "articles" not in data:
            return pd.DataFrame(), f"⚠️ No news found for '{keyword}'."

        if len(data.get("articles")) == 0:
            return pd.DataFrame(), f"⚠️ No results for '{keyword}'."

        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })

        return pd.DataFrame(rows), None

    except Exception as e:
        return pd.DataFrame(), f"⚠️ Error fetching NewsAPI: {e}"


# =========================
# Emotion Color Table
# =========================
DEFAULT_RGB = {
    "joy": (255,200,60),
    "love": (255,95,150),
    "pride": (190,100,255),
    "hope": (60,235,190),
    "curiosity": (50,190,255),
    "calm": (70,135,255),
    "surprise": (255,160,70),
    "neutral": (190,190,200),
    "sadness": (80,120,230),
    "anger": (245,60,60),
    "fear": (150,70,200),
    "disgust": (150,200,60),
    "anxiety": (255,200,60),
    "boredom": (135,135,145),
    "nostalgia": (250,210,150),
    "gratitude": (90,230,230),
    "awe": (120,245,255),
    "trust": (60,200,160),
    "confusion": (255,140,180),
    "mixed": (230,190,110),
}

COLOR_NAMES = {
    "joy":"Jupiter Gold", "love":"Rose", "pride":"Violet", "hope":"Mint",
    "curiosity":"Azure", "calm":"Indigo", "surprise":"Peach", "neutral":"Gray",
    "sadness":"Ocean", "anger":"Vermilion", "fear":"Mulberry", "disgust":"Olive",
    "anxiety":"Sand", "boredom":"Slate", "nostalgia":"Cream", "gratitude":"Cyan",
    "awe":"Ice", "trust":"Teal", "confusion":"Blush", "mixed":"Amber"
}


# =========================
# Sentiment → Emotion Mapping
# =========================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0, "neu":1.0, "pos":0.0, "compound":0.0}
    return sia.polarity_scores(text)


def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5: return "surprise"

    if comp <= -0.65 and neg > 0.5: return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45: return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"

    if neg > 0.5 and neu > 0.3: return "anxiety"
    if neg > 0.45 and pos < 0.1: return "disgust"

    if neu > 0.75 and abs(comp) < 0.1: return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"

    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    return "neutral"


# =========================
# Palette State Management
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}


def get_active_palette():
    """CSV override mode OR default+custom merge"""
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))

    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged


def add_custom_emotion(name, r, g, b):
    if not name:
        return
    st.session_state["custom_palette"][name.strip()] = (int(r), int(g), int(b))
# ============================================================
# Emotional Crystal — FINAL VERSION (Part 3 / 6)
# Crystal Shapes + Renderer + Color Helpers
# ============================================================

import math


# =========================
# Color Helpers
# =========================
def _rgb01(rgb):
    """Convert [0-255] RGB → [0-1] float RGB"""
    c = np.array(rgb, dtype=np.float32) / 255.0
    return np.clip(c, 0, 1)


def vibrancy_boost(rgb, sat_boost=1.28, min_luma=0.38):
    """
    Boost saturation + ensure minimum luma
    Makes crystal colors cinematic & glowing
    """
    c = _rgb01(rgb)
    luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    # Raise dark colors
    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)

    # Apply saturation boost
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    c = np.clip(lum + (c - lum) * sat_boost, 0, 1)

    return tuple(c)


def jitter_color(rgb01, rng, amount=0.06):
    """
    Add small hue variations for natural ice edges
    """
    j = (rng.random(3) - 0.5) * 2 * amount
    c = np.clip(np.array(rgb01) + j, 0, 1)
    return tuple(c.tolist())


# =========================
# Crystal Shape Generator (❄️ NEW)
# =========================
def crystal_shape(center=(0.5, 0.5), r=150, wobble=0.25,
                  sides_min=5, sides_max=10, rng=None):
    """
    Generates a single irregular crystal polygon
    """
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center

    # Random number of sides
    n_vertices = int(rng.integers(sides_min, sides_max + 1))

    # Randomized angle order
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    rng.shuffle(angles)

    # Randomized radius for each vertex
    radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))

    pts = []
    for a, rr in zip(angles, radii):
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((float(x), float(y)))

    # Close polygon
    pts.append(pts[0])

    return pts


# =========================
# Soft Polygon Drawing
# =========================
def draw_polygon_soft(canvas_rgba, pts, color01,
                      fill_alpha=200, blur_px=6, edge_width=0):
    """
    Draws a soft, glowing polygon on the RGBA canvas
    """
    W, H = canvas_rgba.size

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")

    # Convert color
    col = (
        int(color01[0] * 255),
        int(color01[1] * 255),
        int(color01[2] * 255),
        int(fill_alpha)
    )

    # Fill polygon
    d.polygon(pts, fill=col)

    # Optional glowing white outline
    if edge_width > 0:
        edge = (255, 255, 255, max(80, fill_alpha // 2))
        d.line(pts, fill=edge, width=edge_width, joint="curve")

    # Blur layer
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

    # Composite
    canvas_rgba.alpha_composite(layer)


# =========================
# Crystal Renderer (❄️ CORE)
# =========================
def render_crystalmix(
    df, palette,
    width=1500, height=850,
    seed=12345,
    shapes_per_emotion=10,
    min_size=60, max_size=220,
    fill_alpha=210,
    blur_px=6,
    bg_color=(0, 0, 0),
    wobble=0.25,
    layers=10
):
    """
    Main crystal mix renderer
    """

    rng = np.random.default_rng(seed)

    # Background
    base = Image.new("RGBA", (width, height),
                     (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Emotions present in df
    emotions = df["emotion"].value_counts().index.tolist()

    if not emotions:
        emotions = ["neutral"]

    # Multi-layer rendering
    for _layer in range(layers):

        for emo in emotions:

            # Pick color
            base_rgb = palette.get(emo, palette.get("mixed", (230,190,110)))
            base01 = vibrancy_boost(base_rgb, sat_boost=1.30, min_luma=0.40)

            # Draw many shapes for the emotion
            for _ in range(int(shapes_per_emotion)):

                cx = rng.uniform(0.05 * width, 0.95 * width)
                cy = rng.uniform(0.08 * height, 0.92 * height)

                size = int(rng.uniform(min_size, max_size))

                pts = crystal_shape(
                    center=(cx, cy),
                    r=size,
                    wobble=wobble,
                    sides_min=5,
                    sides_max=10,
                    rng=rng
                )

                col01 = jitter_color(base01, rng, amount=0.07)

                local_alpha = int(
                    np.clip(fill_alpha * rng.uniform(0.85, 1.05), 40, 255)
                )
                local_blur = int(np.clip(blur_px * rng.uniform(0.7, 1.4), 0, 40))

                # Some shapes have glowing edges
                edge_w = 0 if rng.random() < 0.6 else max(1, int(size * 0.02))

                draw_polygon_soft(
                    canvas, pts, col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    # Combine layers onto background
    base.alpha_composite(canvas)
    return base.convert("RGB")
# ============================================================
# Emotional Crystal — FINAL VERSION (Part 4 / 6)
# Cinematic Color System
# ============================================================

# =========================
# sRGB ↔ Linear
# =========================
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, x * 12.92,
                    1.055 * (x ** (1 / 2.4)) - 0.055)


# =========================
# Filmic Tonemap (Hable)
# =========================
def filmic_tonemap(x):
    """
    Hable curve for cinematic highlight handling
    """
    A = 0.22
    B = 0.30
    C = 0.10
    D = 0.20
    E = 0.01
    F = 0.30

    return ((x * (A * x + C * B) + D * E) /
            (x * (A * x + B) + D * F)) - E / F


# =========================
# White Balance
# =========================
def apply_white_balance(lin_img, temp, tint):
    """
    lin_img: linear RGB image
    temp: [-1, 1] (blue ↔ yellow)
    tint: [-1, 1] (green ↔ magenta)
    """
    temp_strength = 0.6
    tint_strength = 0.5

    # Temperature shift
    wb_temp = np.array([
        1.0 + temp * temp_strength,   # R
        1.0,                          # G
        1.0 - temp * temp_strength    # B
    ])

    # Tint shift
    wb_tint = np.array([
        1.0 + tint * tint_strength,   # R
        1.0 - tint * tint_strength,   # G
        1.0 + tint * tint_strength    # B
    ])

    wb = wb_temp * wb_tint

    out = lin_img * wb.reshape(1, 1, 3)
    return np.clip(out, 0, 4)


# =========================
# Basic Adjustments
# =========================
def adjust_contrast(img, c):
    return np.clip((img - 0.5) * c + 0.5, 0, 1)


def adjust_saturation(img, s):
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    lum = lum[..., None]
    return np.clip(lum + (img - lum) * s, 0, 1)


def gamma_correct(img, g):
    return np.clip(img ** (1.0 / g), 0, 1)


# =========================
# Highlight Roll-off
# =========================
def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5)
    threshold = 0.8

    mask = np.clip((img - threshold) / (1e-6 + 1.0 - threshold), 0, 1)

    out = img * (1 - mask) + \
          (threshold + (img - threshold) / (1.0 + 4.0 * t * mask)) * mask

    return np.clip(out, 0, 1)


# =========================
# Split Toning
# =========================
def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126 * img[:, :, 0] + \
          0.7152 * img[:, :, 1] + \
          0.0722 * img[:, :, 2]

    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    shadows = np.clip(1.0 - lum + 0.5 * (1 - balance), 0, 1)[..., None]
    highlights = np.clip(lum + 0.5 * (1 + balance) - 0.5, 0, 1)[..., None]

    sh_col = np.array(sh_rgb).reshape(1, 1, 3)
    hi_col = np.array(hi_rgb).reshape(1, 1, 3)

    out = img + shadows * sh_col * 0.25 + highlights * hi_col * 0.25
    return np.clip(out, 0, 1)


# =========================
# Bloom
# =========================
def apply_bloom(img, radius=6.0, intensity=0.6):
    pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

    if radius > 0:
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        blur_arr = np.array(blurred).astype(np.float32) / 255.0
        return np.clip(img * (1 - intensity) + blur_arr * intensity, 0, 1)

    return img


# =========================
# Vignette
# =========================
def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]

    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)

    dist = np.sqrt(xx * xx + yy * yy)
    mask = np.clip(1 - strength * (dist ** 1.5), 0, 1)

    return np.clip(img * mask[..., None], 0, 1)


# =========================
# Ensure Colorfulness
# =========================
def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)

    sat = (maxc - minc) / (maxc + 1e-6)

    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)

    return img


# =========================
# Auto Brightness Compensation (ABC)
# =========================
def auto_brightness_compensation(
    img_arr, target_mean=0.50, strength=0.9,
    black_point_pct=0.05, white_point_pct=0.997,
    max_gain=2.6
):
    """
    Automatically adjusts:
    - black point
    - white point
    - luminance gain
    - preserves cinematic tonality
    """
    arr = np.clip(img_arr, 0, 1).astype(np.float32)

    # Convert to linear
    lin = srgb_to_linear(arr)

    # Luma channel
    Y = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)

    if wp <= bp + 1e-6:
        wp = bp + 1e-3

    # Normalize black/white
    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    Y_final = (1 - strength) * Y + strength * Y_remap

    # Luminance gain
    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0 / max_gain, max_gain)

    lin *= gain

    # Blend remapped & gained
    Y2 = 0.2126 * lin[:, :, 0] + \
         0.7152 * lin[:, :, 1] + \
         0.0722 * lin[:, :, 2]

    blend = 0.65 * strength
    Y_mix = (1 - blend) * Y2 + blend * np.clip(Y_final * gain, 0, 2.5)

    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[..., None], 0, 4)

    # Filmic → back to sRGB
    out = filmic_tonemap(np.clip(lin, 0, 4))
    out = np.clip(out, 0, 1)
    out = linear_to_srgb(out)

    return np.clip(out, 0, 1)
# ============================================================
# Emotional Crystal — FINAL VERSION (Part 5 / 6)
# Sidebar UI + Crystal Engine Controls
# ============================================================

# =========================
# Sidebar — Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")

keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora, emotion)",
    value=st.session_state.get("keyword", ""),
    placeholder="Enter keyword..."
)

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate (All Random Mode)")  # ❄️ FULL RANDOM


# =========================
# Sidebar — Emotion Filter
# =========================
st.sidebar.header("2) Emotion Filter")

cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0,
                            st.session_state.get("cmp_min", -1.0), 0.01)
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0,
                            st.session_state.get("cmp_max", 1.0), 0.01)

init_palette_state()
active_palette = get_active_palette()

# Only emotions appearing in df will show
available_emotions = sorted(df["emotion"].unique().tolist())

def _emo_label(e):
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = active_palette.get(e, (0,0,0))
    return f"{e} (Custom {r},{g},{b})"

auto_top3 = st.sidebar.checkbox("Auto-select Top 3 Emotions", True)

if auto_top3:
    top3 = df["emotion"].value_counts().head(3).index.tolist()
else:
    top3 = available_emotions

selected_labels = st.sidebar.multiselect(
    "Selected Emotions",
    [_emo_label(e) for e in available_emotions],
    [_emo_label(e) for e in top3]
)

selected_emotions = [s.split(" (")[0] for s in selected_labels]


# Apply emotion filters to df
df = df[(df["emotion"].isin(selected_emotions)) &
        (df["compound"] >= cmp_min) &
        (df["compound"] <= cmp_max)]


# =========================
# Sidebar — Crystal Engine
# =========================
st.sidebar.header("3) Crystal Engine")

ribbons_per_emotion = st.sidebar.slider(
    "Crystals per Emotion", 1, 40,
    st.session_state.get("ribbons_per_emotion", 10)
)

stroke_blur = st.sidebar.slider(
    "Crystal Softness (blur px)", 0.0, 20.0,
    st.session_state.get("stroke_blur", 6.0)
)

ribbon_alpha = st.sidebar.slider(
    "Crystal Alpha", 40, 255,
    st.session_state.get("ribbon_alpha", 210)
)

# Crystal size
st.sidebar.subheader("Crystal Size")

poly_min_size = st.sidebar.slider(
    "Min Size (px)", 20, 300,
    st.session_state.get("poly_min_size", 70)
)

poly_max_size = st.sidebar.slider(
    "Max Size (px)", 60, 600,
    st.session_state.get("poly_max_size", 220)
)


# =========================
# Sidebar — Crystal Layers & Randomness
# =========================
st.sidebar.subheader("Crystal Layer Controls")

layer_count = st.sidebar.slider(
    "Layers", 1, 30,
    st.session_state.get("layers", 2)
)

wobble_control = st.sidebar.slider(
    "Wobble (Shape Randomness)",
    0.0, 1.0,
    st.session_state.get("wobble", 0.25),
    step=0.01
)

seed_control = st.sidebar.slider(
    "Seed (Crystal Randomness)",
    0, 500,
    st.session_state.get("seed_control", 25)
)


# =========================
# Sidebar — Background Color
# =========================
st.sidebar.subheader("Background Color")

def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0,2,4))

bg_hex = st.sidebar.color_picker(
    "Choose Background",
    value=st.session_state.get("bg_custom", "#000000")
)

bg_rgb = _hex_to_rgb(bg_hex)


# =========================
# Sidebar — Cinematic Color System
# =========================
st.sidebar.header("4) Cinematic Color")

exp = st.sidebar.slider("Exposure (stops)", -0.2, 1.8,
                        st.session_state.get("exp", 0.55), 0.01)

contrast = st.sidebar.slider("Contrast", 0.70, 1.80,
                             st.session_state.get("contrast", 1.18))

saturation = st.sidebar.slider("Saturation", 0.70, 1.90,
                               st.session_state.get("saturation", 1.18))

gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40,
                              st.session_state.get("gamma_val", 0.92))

roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50,
                         st.session_state.get("roll", 0.40))


# White Balance
st.sidebar.subheader("White Balance")

temp = st.sidebar.slider(
    "Temperature (Blue ↔ Yellow)",
    -1.0, 1.0,
    st.session_state.get("temp", 0.00)
)

tint = st.sidebar.slider(
    "Tint (Green ↔ Magenta)",
    -1.0, 1.0,
    st.session_state.get("tint", 0.00)
)


# Split Toning
st.sidebar.subheader("Split Toning")

sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", 0.08))
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", 0.06))
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", 0.16))

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", 0.10))
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", 0.08))
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", 0.06))

tone_balance = st.sidebar.slider(
    "Balance (Shadows ↔ Highlights)",
    -1.0, 1.0,
    st.session_state.get("tone_balance", 0.0)
)


# =========================
# Sidebar — Bloom / Vignette
# =========================
st.sidebar.subheader("Bloom & Vignette")

bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
                                 st.session_state.get("bloom_radius", 7.0))

bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
                                   st.session_state.get("bloom_intensity", 0.40))

vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
                                      st.session_state.get("vignette_strength", 0.16))


# =========================
# Sidebar — Auto Brightness
# =========================
st.sidebar.header("5) Auto Brightness Compensation")

auto_bright = st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright", True)
)

target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70,
                                st.session_state.get("target_mean", 0.52))

abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0,
                                 st.session_state.get("abc_strength", 0.92))

abc_black = st.sidebar.slider("Black Point %", 0.00, 0.20,
                              st.session_state.get("abc_black", 0.05))

abc_white = st.sidebar.slider("White Point %", 0.80, 1.00,
                              st.session_state.get("abc_white", 0.997))

abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0,
                                st.session_state.get("abc_max_gain", 2.6))


# =========================
# Sidebar — Palette Controls
# =========================
st.sidebar.header("6) Custom Palette (RGB)")

use_csv = st.sidebar.checkbox(
    "Use CSV Palette Only",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

with st.sidebar.expander("Add Custom Emotion"):
    col1, col2, col3, col4 = st.columns([1.6, 1, 1, 1])

    emo_name = col1.text_input("Emotion Name")
    r = col2.number_input("R", 0, 255, 180)
    g = col3.number_input("G", 0, 255, 180)
    b = col4.number_input("B", 0, 255, 200)

    if st.button("Add Color"):
        add_custom_emotion(emo_name, r, g, b)

# CSV Import/Export
with st.sidebar.expander("Import / Export Palette CSV"):
    csv_file = st.file_uploader("Import CSV", type=["csv"])
    if csv_file:
        import_palette_csv(csv_file)

    pal = get_active_palette()
    df_pal = pd.DataFrame([
        {"emotion": k, "r": v[0], "g": v[1], "b": v[2]}
        for k, v in pal.items()
    ])

    st.dataframe(df_pal, use_container_width=True)

    dl = export_palette_csv(pal)
    st.download_button(
        "Download CSV",
        data=dl,
        file_name="palette.csv",
        mime="text/csv"
    )


# =========================
# Reset Button
# =========================
st.sidebar.header("7) Reset All")

if st.sidebar.button("Reset App", type="primary"):
    reset_all()
# ============================================================
# Emotional Crystal — FINAL VERSION (Part 6 / 6)
# Main Rendering + Color Pipeline + Download + Data Table
# ============================================================

left, right = st.columns([0.60, 0.40])

# =========================
# LEFT — Crystal Image
# =========================
with left:
    st.subheader("❄️ Crystal Mix Visualization")

    active_palette = get_active_palette()

    # ❄️ Render Crystal
    img = render_crystalmix(
        df=df,
        palette=active_palette,
        width=1500,
        height=850,
        seed=rng_seed,                     # ← global RNG seed (Random/Keyword updated)
        shapes_per_emotion=ribbons_per_emotion,
        min_size=poly_min_size,
        max_size=poly_max_size,
        fill_alpha=int(ribbon_alpha),
        blur_px=int(stroke_blur),
        bg_color=bg_rgb,
        wobble=wobble_control,
        layers=layer_count
    )

    # ==========================================
    # Convert PIL → NumPy float32
    # ==========================================
    arr = np.array(img).astype(np.float32) / 255.0

    # Convert to linear RGB
    lin = srgb_to_linear(arr)

    # ------ Exposure ------
    lin = lin * (2.0 ** exp)

    # ------ White Balance ------
    lin = apply_white_balance(lin, temp, tint)

    # ------ Highlight Roll-off ------
    lin = highlight_rolloff(lin, roll)

    # Back to sRGB
    arr = linear_to_srgb(np.clip(lin, 0, 4))

    # ------ Filmic Tone Mapping ------
    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)

    # ------ Contrast / Saturation / Gamma ------
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    # ------ Split Toning ------
    arr = split_tone(
        arr,
        sh_rgb=(sh_r, sh_g, sh_b),
        hi_rgb=(hi_r, hi_g, hi_b),
        balance=tone_balance
    )

    # ------ Auto Brightness Compensation ------
    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    # ------ Bloom / Vignette ------
    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)

    # ------ Last safeguard for saturation ------
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    # Convert back to PIL
    final_img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")

    # Temporary buffer for download
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    st.image(final_img, use_column_width=True)

    # Download Button
    st.download_button(
        "💾 Download PNG",
        data=buf,
        file_name="crystal_mix.png",
        mime="image/png"
    )


# =========================
# RIGHT — Data Table
# =========================
with right:
    st.subheader("📊 Data & Emotion Mapping")

    df_show = df.copy()

    # Show readable emotion name
    df_show["emotion_display"] = df_show["emotion"].apply(
        lambda e: f"{e} ({COLOR_NAMES.get(e, 'Custom')})"
    )

    cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]

    if "timestamp" in df_show.columns:
        cols.insert(1, "timestamp")
    if "source" in df_show.columns:
        cols.insert(2, "source")

    st.dataframe(df_show[cols], use_container_width=True, height=600)

# -----------------------------
# END OF APP FINAL
# -----------------------------
