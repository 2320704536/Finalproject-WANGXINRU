# ============================================================
# Emotional Crystal — FINAL VERSION C (Fully Corrected CSV Mode)
# CSV-only = TRUE COLOR, NO emotion logic, NO fallback, NO yellow
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

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Crystal — Final C", page_icon="❄️", layout="wide")
st.title("❄️ Emotional Crystal — Final Version C")

# Instructions section
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

**This project transforms text (or random seeds) into cinematic generative ice-crystal visuals.**

### CSV-only Mode (IMPORTANT)
When **Use CSV palette only** is turned on:

- All detected emotions (anger, awe, fear…) will be completely ignored  
- All rows in the DataFrame will be remapped to the names from the CSV  
- Colors will be 100% exactly as CSV specifies  
- No fallback, no yellow, no vibrancy, no jitter  
- PostFX is still enabled (version C)

This guarantees:  
**ANY input → ALWAYS your CSV colors only.**

---

### Workflow
1. Fetch News or click **Random Generate**
2. Load your CSV color palette  
3. Turn on **Use CSV palette only**
4. Render → Export PNG  
""")
# =========================
# VADER
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
# News API
# =========================
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword, "language": "en", "sortBy": "publishedAt",
        "pageSize": page_size, "apiKey": api_key,
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


# =========================
# DEFAULT (will not be used in CSV mode)
# =========================
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


# =========================
# Emotion classification
# (ignored entirely in CSV mode)
# =========================
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


# =========================
# Palette State
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    """Normal mode = DEFAULT + CSV custom  
       CSV-only mode = CSV only"""
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))
    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged


# =========================
# ✔ FINAL FIX: Overwrite df['emotion'] with CSV keys
# =========================
def force_csv_emotions(df):
    """
    Replace all df['emotion'] with CSV palette keys.
    Guarantees:
    - No fallback
    - No yellow
    - No real emotion used
    """
    csv_keys = list(st.session_state.get("custom_palette", {}).keys())
    if not csv_keys:
        return df  # no CSV loaded

    df = df.copy()
    keys = csv_keys
    L = len(keys)
    rows = len(df)

    # assign emotion cycling through CSV keys
    df["emotion"] = [keys[i % L] for i in range(rows)]
    return df
# =========================
# Crystal Shape (NEW)
# =========================
def crystal_shape(center=(0.5, 0.5), r=150, wobble=0.25,
                  sides_min=5, sides_max=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center
    n_vertices = int(rng.integers(sides_min, sides_max + 1))

    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    rng.shuffle(angles)

    radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))

    pts = []
    for a, rr in zip(angles, radii):
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))

    pts.append(pts[0])
    return pts


# =========================
# Soft Crystal Polygon Rendering
# =========================
def draw_polygon_soft(canvas_rgba, pts, color01,
                      fill_alpha=200, blur_px=6, edge_width=0):
    W, H = canvas_rgba.size
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")

    # color01 is 0–1 float
    col = (
        int(color01[0]*255),
        int(color01[1]*255),
        int(color01[2]*255),
        int(fill_alpha)
    )

    d.polygon(pts, fill=col)

    if edge_width > 0:
        edge = (255,255,255,max(80, fill_alpha//2))
        d.line(pts, fill=edge, width=edge_width, joint="curve")

    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

    canvas_rgba.alpha_composite(layer)


# =========================
# Color helpers
# =========================
def _rgb01(rgb):
    return np.clip(np.array(rgb, dtype=np.float32)/255.0, 0, 1)

def vibrancy_boost(rgb, sat_boost=1.28, min_luma=0.38):
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)
    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    return tuple(np.clip(lum + (c - lum)*sat_boost, 0, 1))

def jitter_color(rgb01, rng, amount=0.06):
    j = (rng.random(3)-0.5)*2*amount
    return tuple(np.clip(np.array(rgb01)+j, 0, 1))


# =========================
# Crystal Mix Rendering
# =========================
def render_crystalmix(
    df, palette,
    width=1500, height=850, seed=12345,
    shapes_per_emotion=10,
    min_size=60, max_size=220,
    fill_alpha=210, blur_px=6,
    bg_color=(0,0,0),
    wobble=0.25,
    layers=10
):
    rng = np.random.default_rng(seed)

    base = Image.new("RGBA", (width, height),
                     (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0,0,0,0))

    # IMPORTANT:
    # df['emotion'] already replaced by CSV keys if CSV-only was on
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = list(palette.keys())

    for _layer in range(layers):
        for emo in emotions:

            # ======================
            # 1) COLOR LOGIC
            # ======================
            if st.session_state.get("use_csv_palette", False):
                # CSV-only = exact CSV RGB → no vibrancy, no jitter
                base_rgb = palette[emo]
                base01 = np.array(base_rgb) / 255.0
            else:
                # normal mode
                base_rgb = palette.get(emo, palette.get("mixed", (230,190,110)))
                base01 = vibrancy_boost(base_rgb)

            # ======================
            # 2) Generate fragments
            # ======================
            for _ in range(max(1, int(shapes_per_emotion))):

                cx = rng.uniform(0.05*width, 0.95*width)
                cy = rng.uniform(0.08*height, 0.92*height)
                rr = int(rng.uniform(min_size, max_size))

                pts = crystal_shape(
                    center=(cx, cy),
                    r=rr,
                    wobble=wobble,
                    sides_min=5,
                    sides_max=10,
                    rng=rng
                )

                # ======================
                # 3) Jitter (disabled for CSV)
                # ======================
                if st.session_state.get("use_csv_palette", False):
                    col01 = base01
                else:
                    col01 = jitter_color(base01, rng, amount=0.07)

                local_alpha = int(np.clip(fill_alpha * rng.uniform(0.85,1.05), 40,255))
                local_blur = max(0, int(blur_px * rng.uniform(0.7,1.4)))
                edge_w = 0 if rng.random() < 0.6 else max(1, int(rr*0.02))

                draw_polygon_soft(
                    canvas, pts, col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    base.alpha_composite(canvas)
    return base.convert("RGB")
# =========================
# Cinematic Color System
# =========================

def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, x*12.92, 1.055*(x**(1/2.4)) - 0.055)

def filmic_tonemap(x):
    A = 0.22; B = 0.30; C = 0.10; D = 0.20; E = 0.01; F = 0.30
    return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F


# ================
# White Balance
# ================
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
    out = lin_img * wb.reshape(1,1,3)
    return np.clip(out, 0, 4)


# ================
# Tone Controls
# ================
def adjust_contrast(img, c):
    return np.clip((img - 0.5)*c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    return np.clip(lum[...,None] + (img-lum[...,None])*s, 0, 1)

def gamma_correct(img, g):
    return np.clip(img ** (1.0/g), 0, 1)


# ================
# Highlights
# ================
def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5)
    threshold = 0.8
    mask = np.clip((img - threshold)/(1e-6 + 1.0 - threshold), 0, 1)
    out = img*(1-mask) + (threshold + (img-threshold)/(1.0+4.0*t*mask))*mask
    return np.clip(out, 0, 1)


# ================
# Split Toning
# ================
def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum_norm = (lum - lum.min())/(lum.max() - lum.min() + 1e-6)

    sh_mask = np.clip(1.0 - lum_norm + 0.5*(1-balance), 0, 1)[...,None]
    hi_mask = np.clip(lum_norm + 0.5*(1+balance) - 0.5, 0, 1)[...,None]

    out = np.clip(
        img +
        sh_mask * np.array(sh_rgb).reshape(1,1,3) * 0.25 +
        hi_mask * np.array(hi_rgb).reshape(1,1,3) * 0.25,
        0, 1
    )
    return out


# ================
# Bloom
# ================
def apply_bloom(img, radius=6.0, intensity=0.6):
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8), "RGB")
    if radius > 0:
        blur = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blur).astype(np.float32)/255.0
        return np.clip(img*(1-intensity) + b*intensity, 0, 1)
    return img


# ================
# Vignette
# ================
def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2)/(w/2); yy = (yy - h/2)/(h/2)
    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength*(r**1.5), 0.0, 1.0)
    return np.clip(img * mask[...,None], 0, 1)


# ================
# Colorfulness Guarantee
# ================
def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)

    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img


# =========================
# Auto Brightness Compensation
# =========================
def auto_brightness_compensation(
    img_arr, target_mean=0.50, strength=0.9,
    black_point_pct=0.05, white_point_pct=0.997,
    max_gain=2.6
):
    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)

    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]

    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-6:
        wp = bp + 1e-3

    Y_remap = np.clip((Y - bp)/(wp - bp), 0, 1)
    remap_gain = np.clip(strength, 0, 1)

    Y_final = (1-remap_gain)*Y + remap_gain*Y_remap

    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)

    lin *= gain

    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    blend = 0.65 * remap_gain

    Y_mix = (1-blend)*Y2 + blend*np.clip(Y_final*gain, 0, 2.5)

    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[...,None], 0, 4)

    out = filmic_tonemap(np.clip(lin,0,4))
    return np.clip(linear_to_srgb(out), 0, 1)
# =========================
# CSV Palette Import
# =========================
def import_palette_csv(file):
    """
    读取 CSV，并强制覆盖 palette。
    CSV-only 模式下，这个 palette 会成为唯一颜色来源。
    """
    try:
        dfc = pd.read_csv(file)

        expected = {"emotion", "r", "g", "b"}
        lower_cols = {c.lower(): c for c in dfc.columns}

        if not expected.issubset(lower_cols.keys()):
            st.error("CSV must contain columns: emotion, r, g, b")
            return

        pal = {}
        for _, row in dfc.iterrows():
            emo = str(row[lower_cols["emotion"]]).strip()
            try:
                r = int(row[lower_cols["r"]])
                g = int(row[lower_cols["g"]])
                b = int(row[lower_cols["b"]])
            except:
                continue

            pal[emo] = (r, g, b)

        # ★ 覆盖 session palette
        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} emotion colors via CSV.")

    except Exception as e:
        st.error(f"CSV Import Error: {e}")


# =========================
# CSV Palette Export
# =========================
def export_palette_csv(pal):
    buf = BytesIO()
    df_out = pd.DataFrame([
        {"emotion": k, "r": v[0], "g": v[1], "b": v[2]}
        for k, v in pal.items()
    ])
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# =========================
# Helpers — Safe Text BBox
# =========================
def safe_text_bbox(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except:
        try:
            return draw.textsize(text, font=font)
        except:
            return 0, 0


# =========================
# Optional Title Overlay
# =========================
def add_title(img_rgb, title, color_rgb=(255,255,255)):
    W, H = img_rgb.size
    rgba = img_rgb.convert("RGBA")
    overlay = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(overlay, "RGBA")

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=int(H*0.06))
    except:
        font = ImageFont.load_default()

    tw, th = safe_text_bbox(d, title, font)
    pad = int(H*0.02)
    x, y = pad, pad

    # 半透明背景条
    bg_rect = [x-10, y-6, x + tw + 16, y + th + 10]
    d.rectangle(bg_rect, fill=(0,0,0,140))
    d.text((x,y), title, font=font, fill=(color_rgb[0],color_rgb[1],color_rgb[2],255))

    rgba.alpha_composite(overlay)
    return rgba.convert("RGB")



# =========================
# DEFAULT CONTROL VALUES
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

    "bg_custom": "#000000"
}


# =========================
# Reset App State
# =========================
def reset_all():
    st.session_state.clear()
    st.rerun()
# =========================
# Sidebar — Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora, science)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
    placeholder="e.g., AI"
)

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate (Crystal Mode)")  # ❄️ NEW


# =========================
# Load Data (Random / Fetch / Fallback)
# =========================
df = pd.DataFrame()

# ❄️ RANDOM MODE
if random_btn:
    st.session_state["auto_seed"] = int(np.random.randint(0, 100000))

    rng = np.random.default_rng()
    num_items = 12
    texts = []
    emos = []

    # 清空 palette，只用 CSV
    st.session_state["custom_palette"] = {}

    for i in range(num_items):
        texts.append(f"Random crystal fragment #{i+1}")
        emo = f"crystal_{i+1}"
        emos.append(emo)

        r = int(rng.integers(0,256))
        g = int(rng.integers(0,256))
        b = int(rng.integers(0,256))

        st.session_state["custom_palette"][emo] = (r,g,b)

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "timestamp": str(date.today()),
        "compound": 0,
        "pos": 0, "neu": 1, "neg": 0,
        "source": "CrystalGen"
    })


# 🌐 FETCH NEWS MODE
elif fetch_btn:
    st.session_state["auto_seed"] = int(np.random.randint(0, 100000))

    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
        df = pd.DataFrame()
    else:
        df = fetch_news(key, keyword if keyword.strip() else "aurora")


# Default data (没有 fetch / 没有 random)
if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the northern sky.",
        "Calm atmospheric conditions create a beautiful environment.",
        "Anxiety rises during unstable market conditions.",
        "A moment of awe as the sky shines green.",
        "Hope emerges with new scientific discoveries."
    ]})
    df["timestamp"] = str(date.today())


# =========================
# Sentiment → Emotion (只在非-random模式)
# =========================
if "emotion" not in df.columns:
    sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================
# Sidebar — Emotion Mapping
# =========================
st.sidebar.header("2) Emotion Mapping")

# Compound range filter
cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]), 0.01)
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]), 0.01)

# Initialize palette
init_palette_state()
base_palette = get_active_palette()

available_emotions = sorted(df["emotion"].unique().tolist())


# =========================
# CSV-only 强制 emotion 覆盖
# =========================
def force_csv_emotions(df, csv_palette):
    """
    使用 CSV 颜色的 emotion keys 重写 df["emotion"]，
    忽略原始 anger/awe/fear 等任何内容。
    """
    csv_keys = list(csv_palette.keys())
    if not csv_keys:
        return df

    n = len(csv_keys)
    df = df.copy()

    # 按顺序循环：emo1, emo2, emo3, ...
    df["emotion"] = [csv_keys[i % n] for i in range(len(df))]
    return df


# CSV-only：强制覆盖
if st.session_state.get("use_csv_palette", False):
    df = force_csv_emotions(df, st.session_state["custom_palette"])
    selected_emotions = df["emotion"].unique().tolist()
else:
    # Normal mode: standard emotion selector UI
    auto_top3 = st.sidebar.checkbox(
        "Auto-select Top-3 emotions",
        value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"])
    )

    top3 = []
    if auto_top3:
        vc = df["emotion"].value_counts()
        top3 = vc.head(3).index.tolist()

    def _label_emotion(e: str) -> str:
        if e in COLOR_NAMES:
            return f"{e} ({COLOR_NAMES[e]})"
        r,g,b = base_palette.get(e, (0,0,0))
        return f"{e} (Custom {r},{g},{b})"

    option_labels = [_label_emotion(e) for e in available_emotions]
    default_labels = [_label_emotion(e) for e in (top3 if top3 else available_emotions)]

    selected_labels = st.sidebar.multiselect(
        "Selected Emotions",
        option_labels,
        default=default_labels
    )
    selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]


# Apply emotion & compound filters
df = df[
    (df["emotion"].isin(selected_emotions)) &
    (df["compound"] >= cmp_min) &
    (df["compound"] <= cmp_max)
]


# =========================
# Sidebar — Crystal Controls
# =========================
st.sidebar.subheader("Crystal Layer Controls")

layer_count = st.sidebar.slider(
    "Layers",
    1, 30,
    2
)

wobble_control = st.sidebar.slider(
    "Wobble (Crystal Randomness)",
    0.00, 1.00,
    0.25,
    step=0.01
)

seed_control = st.sidebar.slider(
    "Seed",
    0, 500,
    st.session_state.get("auto_seed", 25)
)

# Crystal Engine
st.sidebar.header("3) Crystal Engine")

ribbons_per_emotion = st.sidebar.slider(
    "Crystals per Emotion",
    1, 40,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"])
)

stroke_blur = st.sidebar.slider(
    "Crystal Softness (blur px)",
    0.0, 20.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"])
)

ribbon_alpha = st.sidebar.slider(
    "Crystal Alpha",
    40, 255,
    st.session_state.get("ribbon_alpha", DEFAULTS["ribbon_alpha"])
)

# Crystal size
st.sidebar.subheader("Crystal Size Range")
poly_min_size = st.sidebar.slider(
    "Min Size (px)",
    20, 300,
    st.session_state.get("poly_min_size", DEFAULTS["poly_min_size"])
)
poly_max_size = st.sidebar.slider(
    "Max Size (px)",
    60, 600,
    st.session_state.get("poly_max_size", DEFAULTS["poly_max_size"])
)


# =========================
# Background
# =========================
st.sidebar.subheader("Background (Solid Color)")
def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0,2,4))

bg_custom = st.sidebar.color_picker(
    "Choose Background",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"])
)
bg_rgb = _hex_to_rgb(bg_custom)


# =========================
# PostFX (Exposure / WB / Bloom)
# =========================
st.sidebar.header("4) Cinematic Color")

exp = st.sidebar.slider("Exposure (stops)", -0.2, 1.8,
    st.session_state.get("exp", DEFAULTS["exp"]), 0.01
)
contrast = st.sidebar.slider("Contrast", 0.70, 1.80,
    st.session_state.get("contrast", DEFAULTS["contrast"])
)
saturation = st.sidebar.slider("Saturation", 0.70, 1.90,
    st.session_state.get("saturation", DEFAULTS["saturation"])
)
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40,
    st.session_state.get("gamma_val", DEFAULTS["gamma_val"])
)
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50,
    st.session_state.get("roll", DEFAULTS["roll"])
)

# White Balance
st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature", -1.0, 1.0,
    st.session_state.get("temp", DEFAULTS["temp"]))
tint = st.sidebar.slider("Tint", -1.0, 1.0,
    st.session_state.get("tint", DEFAULTS["tint"]))

# Split toning
st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", DEFAULTS["sh_r"]))
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", DEFAULTS["sh_g"]))
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", DEFAULTS["sh_b"]))

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", DEFAULTS["hi_r"]))
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", DEFAULTS["hi_g"]))
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", DEFAULTS["hi_b"]))

tone_balance = st.sidebar.slider(
    "Tone Balance",
    -1.0, 1.0,
    st.session_state.get("tone_balance", DEFAULTS["tone_balance"])
)

# Bloom & Vignette
st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
    st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]))
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
    st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]))
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
    st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]))


# Auto Brightness
st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"])
)
target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70,
    st.session_state.get("target_mean", DEFAULTS["target_mean"]))
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0,
    st.session_state.get("abc_strength", DEFAULTS["abc_strength"]))
abc_black = st.sidebar.slider("Black Point %", 0.00, 0.20,
    st.session_state.get("abc_black", DEFAULTS["abc_black"]))
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00,
    st.session_state.get("abc_white", DEFAULTS["abc_white"]))
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0,
    st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"]))


# =========================
# Sidebar — CSV Palette Switch
# =========================
st.sidebar.header("6) Color Palette")
use_csv = st.sidebar.checkbox(
    "Use CSV palette only (force override)",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)


# =========================
# Sidebar — Reset
# =========================
st.sidebar.header("7) Output Control")
if st.sidebar.button("Reset All", type="primary"):
    reset_all()


# =========================
# Layout: Image Left / Table Right
# =========================
left, right = st.columns([0.60, 0.40])

# =========================
# LEFT — CRYSTAL IMAGE
# =========================
with left:
    st.subheader("❄️ Crystal Mix Visualization")

    # CSV-only: 强制 palette = CSV
    if st.session_state.get("use_csv_palette", False):
        working_palette = dict(st.session_state["custom_palette"])
    else:
        working_palette = get_active_palette()

    # Render crystal
    img = render_crystalmix(
        df=df,
        palette=working_palette,
        width=1500,
        height=850,
        seed=seed_control,
        shapes_per_emotion=ribbons_per_emotion,
        min_size=poly_min_size,
        max_size=poly_max_size,
        fill_alpha=int(ribbon_alpha),
        blur_px=int(stroke_blur),
        bg_color=bg_rgb,
        wobble=wobble_control,
        layers=layer_count
    )

    arr = np.array(img).astype(np.float32) / 255.0

    # CSV-only: 仍保留 PostFX（版本 C）
    # 没有 vibrancy / 没有 jitter / 没有 fallback
    # 所以 arr 是纯 CSV 颜色

    # Linear Space
    lin = srgb_to_linear(arr)
    lin = lin * (2.0 ** exp)
    lin = apply_white_balance(lin, temp, tint)
    lin = highlight_rolloff(lin, roll)
    arr = linear_to_srgb(np.clip(lin, 0, 4))

    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    arr = split_tone(
        arr,
        sh_rgb=(sh_r, sh_g, sh_b),
        hi_rgb=(hi_r, hi_g, hi_b),
        balance=tone_balance
    )

    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    final_img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="RGB")

    # Display & Download
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    st.image(buf, use_column_width=True)
    st.download_button(
        "💾 Download PNG",
        data=buf,
        file_name="crystal_mix_final_C.png",
        mime="image/png"
    )


# =========================
# RIGHT — DATA TABLE
# =========================
with right:
    st.subheader("📊 Data Table")

    df2 = df.copy()
    df2["emotion_display"] = df2["emotion"]

    cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns:
        cols.insert(1, "timestamp")
    if "source" in df.columns:
        cols.insert(2, "source")

    st.dataframe(df2[cols], use_container_width=True, height=600)
