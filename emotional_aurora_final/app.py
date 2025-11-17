# ============================================================
# Emotional Crystal — FINAL FULL VERSION (PART 1 / 4)
# Completely Random Mode + Fixed Emotion Names + CSV Overwrite
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

# ============================================================
# Streamlit App Setup
# ============================================================
st.set_page_config(page_title="Emotional Crystal — Final", page_icon="❄️", layout="wide")
st.title("❄️ Emotional Crystal — Final")

with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

**This project transforms emotion data into cinematic ice-crystal generative visuals.**

**1. Fetch or Generate Data**  
- Enter a keyword  
- Fetch news through **NewsAPI**  
- Or click **Random Generate** for fully generative crystal mode  

**2. Emotion Classification (Keyword Mode)**  
- Text sentiment analyzed using VADER  
- Each entry mapped to one of 20+ emotions  
- Only emotions *present in df* appear in Selected Emotions  

**3. Crystal Rendering**  
- Generates layered **ice-crystal fragments**  
- Each emotion has a unique color  
- Supports cinematic bloom, tonemap, WB, gamma, etc.

**4. Random Mode**  
- **Fully random crystals + fully random colors**  
- But *CSV palette will overwrite these*  
- Random mode **ignores Selected Emotions**, but keeps all sliders  

**5. Cinematic Controls**  
- Exposure / Contrast / Saturation  
- White balance (Temp / Tint)  
- Split toning  
- Auto brightness  
- Bloom + Vignette  

**6. Export Image**  
- Download PNG with one click  
""")


# ============================================================
# Load VADER (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


# ============================================================
# News API
# ============================================================
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

        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()

        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()


# ============================================================
# Default Emotion Colors
# ============================================================
DEFAULT_RGB = {
    "joy":        (255,200,60),
    "love":       (255,95,150),
    "pride":      (190,100,255),
    "hope":       (60,235,190),
    "curiosity":  (50,190,255),
    "calm":       (70,135,255),
    "surprise":   (255,160,70),
    "neutral":    (190,190,200),
    "sadness":    (80,120,230),
    "anger":      (245,60,60),
    "fear":       (150,70,200),
    "disgust":    (150,200,60),
    "anxiety":    (255,200,60),
    "boredom":    (135,135,145),
    "nostalgia":  (250,210,150),
    "gratitude":  (90,230,230),
    "awe":        (120,245,255),
    "trust":      (60,200,160),
    "confusion":  (255,140,180),
    "mixed":      (230,190,110),
}

COLOR_NAMES = {
    "joy":"Jupiter Gold","love":"Rose","pride":"Violet","hope":"Mint",
    "curiosity":"Azure","calm":"Indigo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber"
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())


# ============================================================
# Sentiment → Emotion
# ============================================================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
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
# ============================================================
# Emotional Crystal — FINAL FULL VERSION (PART 2 / 4)
# Crystal Generator + Cinematic Color System
# ============================================================

# ============================================================
# Color Utility Helpers
# ============================================================
def _rgb01(rgb):
    """Convert 0–255 RGB into 0–1 float RGB."""
    c = np.array(rgb, dtype=np.float32) / 255.0
    return np.clip(c, 0, 1)


def vibrancy_boost(rgb, sat_boost=1.25, min_luma=0.40):
    """Ensure colors stay bright enough for cinematic look."""
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]

    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)

    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    c = np.clip(lum + (c - lum) * sat_boost, 0, 1)

    return tuple(c)


def jitter_color(rgb01, rng, amount=0.07):
    """Add slight jitter to color for organic variation."""
    j = (rng.random(3) - 0.5) * 2 * amount
    c = np.clip(np.array(rgb01) + j, 0, 1)
    return tuple(c.tolist())


# ============================================================
# Crystal Shape Generator
# ============================================================
def crystal_shape(center=(0.5, 0.5), r=150,
                  wobble=0.25, sides_min=5, sides_max=10, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center
    n_vertices = int(rng.integers(sides_min, sides_max + 1))

    # Randomized angular distribution
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    rng.shuffle(angles)

    # Random radius per vertex
    radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))

    pts = []
    for a, rr in zip(angles, radii):
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((float(x), float(y)))

    pts.append(pts[0])  # close polygon
    return pts


# ============================================================
# Soft Polygon Rendering
# ============================================================
def draw_polygon_soft(canvas_rgba, pts, color01,
                      fill_alpha=200, blur_px=6, edge_width=0):

    W, H = canvas_rgba.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")

    # Fill color
    col = (
        int(color01[0] * 255),
        int(color01[1] * 255),
        int(color01[2] * 255),
        int(fill_alpha)
    )

    d.polygon(pts, fill=col)

    # Optional white edge
    if edge_width > 0:
        edge = (255, 255, 255, max(80, fill_alpha // 2))
        d.line(pts, fill=edge, width=edge_width, joint="curve")

    # Soft blur
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

    canvas_rgba.alpha_composite(layer)


# ============================================================
# Crystal Mix Renderer (supports Random Mode)
# ============================================================
def render_crystalmix(
    df,
    palette,
    width=1500,
    height=850,
    seed=12345,
    shapes_per_emotion=10,
    min_size=60,
    max_size=220,
    fill_alpha=210,
    blur_px=6,
    bg_color=(0, 0, 0),
    wobble=0.25,
    layers=10,
    force_random_colors=False   # 🔥关键：Random Mode 强制使用随机颜色
):
    rng = np.random.default_rng(seed)

    base = Image.new("RGBA", (width, height),
                     (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    emotions = df["emotion"].tolist()
    if not emotions:
        emotions = ["fallback1"]

    # ============================================================
    # Random Mode：不受 Selected Emotions 限制
    # 但仍会被 CSV palette 覆盖
    # ============================================================
    random_colors = {}
    if force_random_colors:
        for emo in emotions:
            random_colors[emo] = (
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
            )

    # ============================================================
    # Draw all layers
    # ============================================================
    for _layer in range(layers):
        for emo in emotions:

            # -----------------------------
            # 选择颜色来源规则：
            # 1) Random Mode → 优先 random_colors
            # 2) CSV palette → 会覆盖
            # 3) Default palette → fallback
            # -----------------------------
            if force_random_colors:
                base_rgb = random_colors.get(emo, (200, 200, 200))
            else:
                base_rgb = palette.get(emo, (200, 200, 200))

            # 如果 CSV 覆盖了，则强制使用 CSV 值
            if emo in palette:
                base_rgb = palette[emo]

            base01 = vibrancy_boost(base_rgb, sat_boost=1.25, min_luma=0.40)

            for _ in range(int(shapes_per_emotion)):
                cx = rng.uniform(0.06 * width, 0.94 * width)
                cy = rng.uniform(0.08 * height, 0.92 * height)

                rr = int(rng.uniform(min_size, max_size))

                pts = crystal_shape(
                    center=(cx, cy),
                    r=rr,
                    wobble=wobble,
                    sides_min=5,
                    sides_max=10,
                    rng=rng
                )

                col01 = jitter_color(base01, rng, amount=0.06)
                local_alpha = int(np.clip(fill_alpha * rng.uniform(0.8, 1.05), 30, 255))
                local_blur = int(blur_px * rng.uniform(0.7, 1.4))
                edge_w = 0 if rng.random() < 0.7 else max(1, int(rr * 0.02))

                draw_polygon_soft(
                    canvas,
                    pts,
                    col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    base.alpha_composite(canvas)
    return base.convert("RGB")


# ============================================================
# Cinematic Color Pipeline
# ============================================================
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308,
                    x * 12.92,
                    1.055 * (x ** (1 / 2.4)) - 0.055)


def filmic_tonemap(x):
    A = 0.22; B = 0.30; C = 0.10; D = 0.20; E = 0.01; F = 0.30
    return ((x * (A * x + C * B) + D * E)
            / (x * (A * x + B) + D * F)) - E / F


def apply_white_balance(lin_img, temp, tint):

    temp_strength = 0.6
    tint_strength = 0.5

    wb_temp = np.array([
        1.0 + temp * temp_strength,
        1.0,
        1.0 - temp * temp_strength
    ])

    wb_tint = np.array([
        1.0 + tint * tint_strength,
        1.0 - tint * tint_strength,
        1.0 + tint * tint_strength
    ])

    wb = wb_temp * wb_tint

    out = lin_img * wb.reshape(1, 1, 3)
    return np.clip(out, 0, 4)


def adjust_contrast(img, c):
    return np.clip((img - 0.5) * c + 0.5, 0, 1)


def adjust_saturation(img, s):
    lum = 0.2126 * img[:, :, 0] + \
          0.7152 * img[:, :, 1] + \
          0.0722 * img[:, :, 2]
    lum = lum[..., None]
    return np.clip(lum + (img - lum) * s, 0, 1)


def gamma_correct(img, g):
    return np.clip(img ** (1.0 / g), 0, 1)


def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5)
    threshold = 0.8
    mask = np.clip((img - threshold) / (1e-6 + 1.0 - threshold), 0, 1)
    out = img * (1 - mask) + \
          (threshold + (img - threshold) / (1 + 4 * t * mask)) * mask
    return np.clip(out, 0, 1)


def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126 * img[:, :, 0] + \
          0.7152 * img[:, :, 1] + \
          0.0722 * img[:, :, 2]
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    sh = np.clip(1.0 - lum + 0.5 * (1 - balance), 0, 1)[..., None]
    hi = np.clip(lum + 0.5 * (1 + balance) - 0.5, 0, 1)[..., None]

    sh_col = np.array(sh_rgb).reshape(1, 1, 3)
    hi_col = np.array(hi_rgb).reshape(1, 1, 3)

    out = np.clip(img +
                  sh * sh_col * 0.22 +
                  hi * hi_col * 0.22,
                  0, 1)
    return out


def apply_bloom(img, radius=6, intensity=0.6):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32) / 255.0
        return np.clip(img * (1 - intensity) + b * intensity, 0, 1)
    return img


def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2) / (w/2)
    yy = (yy - h/2) / (h/2)
    r = np.sqrt(xx * xx + yy * yy)
    mask = np.clip(1 - strength * (r ** 1.4), 0, 1)
    return np.clip(img * mask[..., None], 0, 1)


def ensure_colorfulness(img, min_sat=0.16, boost=1.20):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img


# ============================================================
# Auto Brightness Compensation
# ============================================================
def auto_brightness_compensation(
    img_arr, target_mean=0.50, strength=0.9,
    black_point_pct=0.05, white_point_pct=0.997, max_gain=2.6):

    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)

    Y = 0.2126 * lin[:, :, 0] + \
        0.7152 * lin[:, :, 1] + \
        0.0722 * lin[:, :, 2]

    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-6:
        wp = bp + 1e-3

    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    remap_gain = np.clip(strength, 0, 1)
    Y_final = (1 - remap_gain) * Y + remap_gain * Y_remap

    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0 / max_gain, max_gain)
    lin *= gain

    Y2 = 0.2126*lin[:, :, 0] + \
         0.7152*lin[:, :, 1] + \
         0.0722*lin[:, :, 2]

    blend = 0.65 * remap_gain
    Y_mix = (1 - blend)*Y2 + blend*np.clip(Y_final * gain, 0, 2.5)

    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[..., None], 0, 4)

    out = filmic_tonemap(np.clip(lin, 0, 4))
    out = linear_to_srgb(out)
    return np.clip(out, 0, 1)
# ============================================================
# Emotional Crystal — FINAL FULL VERSION (PART 3 / 4)
# Sidebar UI + Data Loading (Keyword / Random) + Emotion Mapping
# ============================================================

# =========================
# Default Values
# =========================
DEFAULTS = {
    "keyword": "",
    "ribbons_per_emotion": 10,
    "stroke_blur": 6.0,
    "ribbon_alpha": 210,

    "poly_min_size": 70,
    "poly_max_size": 220,

    "auto_bright": True,
    "target_mean": 0.52,
    "abc_strength": 0.92,
    "abc_black": 0.05,
    "abc_white": 0.997,
    "abc_max_gain": 2.6,

    "exp": 0.55,
    "contrast": 1.18,
    "saturation": 1.18,
    "gamma_val": 0.92,
    "roll": 0.40,

    "temp": 0.00,
    "tint": 0.00,

    "sh_r": 0.08, "sh_g": 0.06, "sh_b": 0.16,
    "hi_r": 0.10, "hi_g": 0.08, "hi_b": 0.06,
    "tone_balance": 0.0,

    "vignette_strength": 0.16,
    "bloom_radius": 7.0,
    "bloom_intensity": 0.40,

    "cmp_min": -1.0,
    "cmp_max": 1.0,
    "auto_top3": True,

    "bg_custom": "#000000",

    # NEW — selection cache
    "seed_option": 2,     # 1=B, 2=A, 3=A (你要求的)
}


# ============================================================
# Reset Session
# ============================================================
def reset_all():
    st.session_state.clear()
    st.rerun()


# ============================================================
# Sidebar — Data Source
# ============================================================
st.sidebar.header("1) Data Source (NewsAPI)")

keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora, science)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
    placeholder="e.g., AI"
)

fetch_btn = st.sidebar.button("Fetch News (Keyword Mode)")
random_btn = st.sidebar.button("Random Generate (Crystal Mode)")  # 🔥 NEW


# ============================================================
# LOAD DATA (Fetch / Random / Fallback)
# ============================================================
df = pd.DataFrame()

# ============================================================
# ❄ RANDOM MODE — Fully Random Emotions + Colors
# ============================================================
if random_btn:

    rng = np.random.default_rng()

    num_items = rng.integers(10, 18)      # 随机数量（更自然）
    texts = []
    emos = []

    st.session_state["custom_palette"] = {}   # 清空旧 palette

    for i in range(num_items):

        # completely random text
        texts.append(f"Random Crystal Fragment #{i+1}")

        # random emotion name  ——你指定 R1, R2 这种命名方式
        emo = f"R{i+1}"
        emos.append(emo)

        # completely random RGB
        r = int(rng.integers(0, 256))
        g = int(rng.integers(0, 256))
        b = int(rng.integers(0, 256))

        # write into custom palette (可被 CSV 覆盖)
        st.session_state["custom_palette"][emo] = (r, g, b)

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "timestamp": str(date.today()),
        "compound": 0,
        "pos": 0, "neu": 1, "neg": 0,
        "source": "Crystal-Random"
    })

    st.session_state["last_mode"] = "random"



# ============================================================
# 🔍 FETCH MODE — Keyword NewsAPI
# ============================================================
elif fetch_btn:

    key = st.secrets.get("NEWS_API_KEY", "")

    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        q = keyword if keyword.strip() else "aurora"
        df = fetch_news(key, q)

        st.session_state["last_mode"] = "fetch"


# ============================================================
# DEFAULT FALLBACK (when entering page)
# ============================================================
if df.empty:
    df = pd.DataFrame({
        "text": [
            "A breathtaking aurora illuminated the northern sky last night.",
            "Calm atmospheric conditions create a beautiful environment.",
            "Anxiety spreads among investors during unstable market conditions.",
            "A moment of awe as the sky shines with green light.",
            "Hope arises as scientific discoveries advance our understanding."
        ],
        "timestamp": str(date.today())
    })

    st.session_state["last_mode"] = "demo"


# ============================================================
# SENTIMENT + EMOTION CLASSIFICATION (Only for fetch/demo mode)
# ============================================================
df["text"] = df["text"].fillna("")

if st.session_state["last_mode"] != "random":

    if "emotion" not in df.columns:

        sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
        df = pd.concat([df.reset_index(drop=True),
                        sent_df.reset_index(drop=True)], axis=1)

        df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# ============================================================
# Sidebar — Emotion Mapping
# ============================================================
st.sidebar.header("2) Emotion Mapping")

cmp_min = st.sidebar.slider(
    "Compound Min", -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]),
    0.01
)

cmp_max = st.sidebar.slider(
    "Compound Max", -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]),
    0.01
)


# -----------------------------
# get palette
# -----------------------------
init_palette_state()
base_palette = get_active_palette()

# -----------------------------
# Available emotions = df only
# -----------------------------
available_emotions = sorted(df["emotion"].unique().tolist())


# Label style
def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = base_palette.get(e, (0, 0, 0))
    return f"{e} (Custom {r},{g},{b})"


auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3",
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"])
)

top3 = []
if auto_top3 and st.session_state["last_mode"] != "random":
    vc = df["emotion"].value_counts()
    top3 = vc.head(3).index.tolist()

option_labels = [_label_emotion(e) for e in available_emotions]
default_labels = [_label_emotion(e) for e in (top3 if top3 else available_emotions)]

selected_labels = st.sidebar.multiselect(
    "Selected Emotions (not applied in Random Mode)",
    option_labels,
    default=default_labels
)

selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]


# ============================================================
# APPLY EMOTION FILTER (Fetch mode only)
# ============================================================
if st.session_state["last_mode"] != "random":

    df = df[
        (df["emotion"].isin(selected_emotions)) &
        (df["compound"] >= cmp_min) &
        (df["compound"] <= cmp_max)
    ]


# ============================================================
# Crystal Layer Controls
# ============================================================
st.sidebar.subheader("Crystal Layers")

layer_count = st.sidebar.slider(
    "Layers", 1, 30, 2,
    help="How many total crystal layers are rendered."
)

wobble_control = st.sidebar.slider(
    "Wobble (shape randomness)",
    0.00, 1.00, 0.25, 0.01
)

seed_control = st.sidebar.slider(
    "Seed",
    0, 999,
    25,
    help="Controls random shape. Smaller = subtle, larger = dramatic."
)
# ============================================================
# Emotional Crystal — FINAL FULL VERSION (PART 4 / 4)
# Main Rendering + Cinematic Pipeline + Display + Download
# ============================================================

left, right = st.columns([0.60, 0.40])

# =========================
# LEFT — Visual Rendering
# =========================
with left:
    st.subheader("❄️ Crystal Mix Visualization")

    working_palette = get_active_palette()

    # ============================
    # ❶ Raw Crystal Rendering
    # ============================
    img = render_crystalmix(
        df=df,
        palette=working_palette,
        width=1500,
        height=850,
        seed=seed_control,                     # ← seed works for both modes
        shapes_per_emotion=ribbons_per_emotion,
        min_size=poly_min_size,
        max_size=poly_max_size,
        fill_alpha=int(ribbon_alpha),
        blur_px=int(stroke_blur),
        bg_color=bg_rgb,
        wobble=wobble_control,
        layers=layer_count
    )

    # ============================
    # ❷ Convert PIL → NumPy
    # ============================
    arr = np.array(img).astype(np.float32) / 255.0
    lin = srgb_to_linear(arr)

    # ============================
    # ❸ Cinematic Exposure
    # ============================
    lin = lin * (2.0 ** exp)

    # ============================
    # ❹ White Balance
    # ============================
    lin = apply_white_balance(lin, temp, tint)

    # ============================
    # ❺ Highlight Roll-off
    # ============================
    lin = highlight_rolloff(lin, roll)

    # ============================
    # ❻ Back to sRGB (pre-filmic)
    # ============================
    arr = linear_to_srgb(np.clip(lin, 0, 4))

    # ============================
    # ❼ Filmic Curve (Hable)
    # ============================
    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)

    # ============================
    # ❽ Contrast / Saturation / Gamma
    # ============================
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    # ============================
    # ❾ Split Toning
    # ============================
    arr = split_tone(
        arr,
        sh_rgb=(sh_r, sh_g, sh_b),
        hi_rgb=(hi_r, hi_g, hi_b),
        balance=tone_balance
    )

    # ============================
    # ❿ Auto Brightness Compensation
    # ============================
    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    # ============================
    # ⓫ Bloom
    # ============================
    arr = apply_bloom(
        arr,
        radius=bloom_radius,
        intensity=bloom_intensity
    )

    # ============================
    # ⓬ Vignette
    # ============================
    arr = apply_vignette(arr, strength=vignette_strength)

    # ============================
    # ⓭ Ensure Colorfulness
    # ============================
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    # ============================
    # ⓮ Convert NumPy → PIL
    # ============================
    final_img = Image.fromarray(
        (np.clip(arr, 0, 1) * 255).astype(np.uint8),
        mode="RGB"
    )

    # Buffer for download
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    # ============================
    # ⓯ Display + Download
    # ============================
    st.image(buf, use_column_width=True)
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

    df2 = df.copy()
    df2["emotion_display"] = df2["emotion"].apply(
        lambda e: f"{e} ({COLOR_NAMES.get(e, 'Custom')})"
    )

    cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]

    if "timestamp" in df.columns:
        cols.insert(1, "timestamp")
    if "source" in df.columns:
        cols.insert(2, "source")

    st.dataframe(df2[cols],
                 use_container_width=True,
                 height=600)
