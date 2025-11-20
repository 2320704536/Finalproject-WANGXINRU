# ============================================================
# Emotional Crystal — Final Hybrid Edition v2025
# (Full Features: NewsAPI, VADER, Emotion Mapping, Crystal Engine, CSV Palette)
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
# Streamlit Page Setup
# ============================================================

st.set_page_config(
    page_title="Emotional Crystal — Final Hybrid",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Emotional Crystal — Final Hybrid Edition")

with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

This version supports both **emotion-based crystal colors** and **CSV palette override**.

#### **CSV-only mode**
- All colors come *only* from CSV  
- Emotions are auto-renamed to **csv_1, csv_2, … in order**  
- All cinematic post-processing remains enabled (your choice)

#### **Hybrid mode**
- Default palette + sentiment mapping + post FX  
- CSV can override or extend colors  
- You may still filter emotions or apply compound thresholds

---
""")

# ============================================================
# Load VADER (once)
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
# NewsAPI Helper
# ============================================================

def fetch_news(api_key, keyword="technology", page_size=50):
    """
    Fetch text data using NewsAPI.
    """
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
# Default Emotion → RGB Palette
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
ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy":"Jupiter Gold","love":"Rose","pride":"Violet","hope":"Mint",
    "curiosity":"Azure","calm":"Indedo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber"
}

# ============================================================
# Initialize Palette State
# ============================================================

def init_palette_state():
    """
    Initialize palette-related session keys.
    """
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False

    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    """
    Hybrid mode = DEFAULT + custom CSV
    CSV-only mode = custom CSV only
    """
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))

    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged
# ============================================================
# Sentiment → Emotion Mapping
# ============================================================

def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row):
    """
    The full rule-based emotion mapping you used.
    Retained because Hybrid version requires default mapping.
    """
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
# CSV Palette Import
# ============================================================

def import_palette_csv(file):
    """
    Import CSV palette and write to session_state["custom_palette"].
    CSV format:
        emotion,r,g,b
    But emotion may be numbers or arbitrary strings.
    """
    try:
        dfc = pd.read_csv(file)

        expected = {"emotion","r","g","b"}
        lower_cols = {c.lower(): c for c in dfc.columns}

        if not expected.issubset(lower_cols.keys()):
            st.error("CSV must contain emotion, r, g, b columns.")
            return

        pal = {}
        for _, row in dfc.iterrows():
            emo_raw = str(row[lower_cols["emotion"]]).strip()

            try:
                r = int(row[lower_cols["r"]])
                g = int(row[lower_cols["g"]])
                b = int(row[lower_cols["b"]])
            except:
                continue

            pal[emo_raw] = (r, g, b)

        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV.")

    except Exception as e:
        st.error(f"CSV import error: {e}")


# ============================================================
# CSV Palette Export
# ============================================================

def export_palette_csv(pal):
    buf = BytesIO()
    df_out = pd.DataFrame([
        {"emotion": k, "r": v[0], "g": v[1], "b": v[2]}
        for k, v in pal.items()
    ])
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ============================================================
# CSV AUTO-RENAME (1A: Strict Sequential Mapping)
# ============================================================

def apply_csv_palette_autorename(df):
    """
    1️⃣ After CSV import:
        custom_palette = {
            "aaa": (..),
            "bbb": (..),
            "ccc": (..)
        }

    2️⃣ Auto rename keys to:
        csv_1, csv_2, csv_3...

    3️⃣ Also rewrite df["emotion"] in **appearance order**
        Example:
            df["emotion"] = ["surprise", "calm", "neutral"]
        becomes:
            df["emotion"] = ["csv_1", "csv_2", "csv_3"]
        according to row appearance, not emotion frequency.

    This perfectly matches your choice 1A.
    """

    if not st.session_state.get("use_csv_palette", False):
        return df   # Only affect CSV-only mode

    pal = st.session_state.get("custom_palette", {})
    if not pal:
        return df

    # 1) Build sequential renaming for palette
    new_palette = {}
    rename_map = {}

    for i, (old_name, rgb) in enumerate(pal.items(), start=1):
        new_name = f"csv_{i}"
        new_palette[new_name] = rgb
        rename_map[old_name] = new_name

    # Replace palette
    st.session_state["custom_palette"] = new_palette

    # 2) Rewrite df["emotion"] by appearance order
    df = df.copy()
    unique_df_emos = df["emotion"].tolist()
    seen = []
    df_map = {}

    for emo in unique_df_emos:
        if emo not in seen:
            seen.append(emo)

    for idx, emo in enumerate(seen, start=1):
        df_map[emo] = f"csv_{idx}"

    df["emotion"] = df["emotion"].apply(lambda e: df_map.get(e, e))

    return df


# ============================================================
# Crystal Shape Generator  (core random geometry)
# ============================================================

def crystal_shape(center=(0.5,0.5), r=150, wobble=0.25,
                  sides_min=5, sides_max=10, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center
    n_vertices = int(rng.integers(sides_min, sides_max+1))

    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    rng.shuffle(angles)

    radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))

    pts = []
    for a, rr in zip(angles, radii):
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((float(x), float(y)))

    pts.append(pts[0])
    return pts


# ============================================================
# Soft Polygon Draw
# ============================================================

def draw_polygon_soft(canvas_rgba, pts, color01,
                      fill_alpha=200, blur_px=6, edge_width=0):

    W, H = canvas_rgba.size
    layer = Image.new("RGBA",(W,H),(0,0,0,0))
    d = ImageDraw.Draw(layer,"RGBA")

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


# ============================================================
# Color Helpers (vibrancy, jitter, etc.)
# ============================================================

def _rgb01(rgb):
    c = np.array(rgb, dtype=np.float32)/255.0
    return np.clip(c,0,1)

def vibrancy_boost(rgb, sat_boost=1.28, min_luma=0.38):
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)
    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    c = np.clip(lum + (c - lum)*sat_boost, 0, 1)
    return tuple(c)

def jitter_color(rgb01, rng, amount=0.06):
    j = (rng.random(3)-0.5)*2*amount
    c = np.clip(np.array(rgb01)+j, 0, 1)
    return tuple(c.tolist())
# ============================================================
# Crystal Mix Renderer — Core Engine
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
    bg_color=(0,0,0),
    wobble=0.25,
    layers=10
):
    """
    Main renderer used by your app.
    Fully fixed for CSV-only mode.
    """

    rng = np.random.default_rng(seed)

    # Background
    base = Image.new("RGBA", (width, height), (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0,0,0,0))

    # All active emotions in data
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        # Very rare fallback
        emotions = list(palette.keys())[:3]

    # Start layering
    for _layer in range(layers):

        for emo in emotions:

            # ====================================================
            # 1) STRICT CSV-ONLY MODE: use only CSV palette colors
            # ====================================================
            if st.session_state.get("use_csv_palette", False):

                # guaranteed to exist because CSV-only
                base_rgb = palette[emo]          # exact (R,G,B)
                base01 = np.array(base_rgb) / 255.0   # exact normalized color
                col01 = base01                    # do NOT change

            else:
            # ====================================================
            # 2) NORMAL MODE: DEFAULT + custom + sensitivity fx
            # ====================================================

                # choose base rgb
                if emo in palette:
                    base_rgb = palette[emo]
                else:
                    # Even in normal mode, remove yellow fallback
                    # Instead fallback to soft neutral gray
                    base_rgb = (180,180,190)

                # vibrancy boost
                base01 = vibrancy_boost(base_rgb, sat_boost=1.30, min_luma=0.40)

                # jitter color (for organic crystal look)
                col01 = jitter_color(base01, rng, amount=0.07)

            # ====================================================
            # 3) For each emotion generate many fragments
            # ====================================================
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

                # local rendering params
                local_alpha = int(
                    np.clip(fill_alpha * rng.uniform(0.85, 1.05), 40, 255)
                )
                local_blur = max(0, int(blur_px * rng.uniform(0.7, 1.4)))
                edge_w = 0 if rng.random() < 0.6 else max(1, int(rr * 0.02))

                draw_polygon_soft(
                    canvas,
                    pts,
                    col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    # Composite all crystals on base
    base.alpha_composite(canvas)
    final = base.convert("RGB")
    return final
# ============================================================
# Cinematic Color System — Corrected v2025
# ============================================================

# ------------ Utility: sRGB ↔ Linear ------------------------
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, x*12.92, 1.055*(x**(1/2.4)) - 0.055)


# ------------ Filmic ACES Curve -----------------------------
def filmic_tonemap(x):
    A = 0.22; B = 0.30; C = 0.10
    D = 0.20; E = 0.01; F = 0.30

    return ((x*(A*x + C*B) + D*E) /
            (x*(A*x + B) + D*F)) - E/F


# ------------ White Balance (Temp & Tint) -------------------
def apply_white_balance(lin_img, temp, tint):
    """
    Temperature (Blue ↔ Yellow)
    Tint (Green ↔ Magenta)
    Both in [-1, 1]
    """

    temp_s = 0.6
    tint_s = 0.5

    wb_temp = np.array([
        1.0 + temp * temp_s,
        1.0,
        1.0 - temp * temp_s
    ])

    wb_tint = np.array([
        1.0 + tint * tint_s,
        1.0 - tint * tint_s,
        1.0 + tint * tint_s
    ])

    wb = wb_temp * wb_tint

    out = lin_img * wb.reshape(1,1,3)
    return np.clip(out, 0, 4)


# ------------------- Basic Adjustments -----------------------
def adjust_contrast(img, c):
    return np.clip((img - 0.5)*c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = lum[...,None]
    return np.clip(lum + (img - lum)*s, 0, 1)

def gamma_correct(img, g):
    return np.clip(img ** (1.0/g), 0, 1)

def highlight_rolloff(img, roll):
    t = np.clip(roll, 0, 1.5)
    threshold = 0.8
    mask = np.clip((img - threshold)/(1e-6 + 1.0 - threshold), 0, 1)
    out = img*(1 - mask) + (threshold + (img-threshold)/(1.0 + 4.0*t*mask))*mask
    return np.clip(out, 0, 1)


# -------------------- Split Toning ---------------------------
def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = (lum - lum.min())/(lum.max()-lum.min() + 1e-6)

    sh = np.clip(1.0 - lum + 0.5*(1 - balance), 0, 1)[...,None]
    hi = np.clip(lum + 0.5*(1 + balance) - 0.5, 0, 1)[...,None]

    sh_col = np.array(sh_rgb).reshape(1,1,3)
    hi_col = np.array(hi_rgb).reshape(1,1,3)

    out = img + sh*sh_col*0.25 + hi*hi_col*0.25
    return np.clip(out, 0, 1)


# ---------------------- Bloom Effect -------------------------
def apply_bloom(img, radius=6.0, intensity=0.6):
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8), mode="RGB")
    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32)/255.0
        out = img*(1-intensity) + b*intensity
        return np.clip(out, 0, 1)
    return img


# ---------------------- Vignette Effect ----------------------
def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2)/(w/2)
    yy = (yy - h/2)/(h/2)
    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength*(r**1.5), 0, 1)
    return np.clip(img * mask[...,None], 0, 1)


# ---------------------- Ensure Colorfulness ------------------
def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)

    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)

    return img


# ============================================================
# Auto Brightness Compensation — With Filmic Safety
# ============================================================

def auto_brightness_compensation(
    img_arr,
    target_mean=0.50,
    strength=0.9,
    black_point_pct=0.05,
    white_point_pct=0.997,
    max_gain=2.6
):
    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)

    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]

    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-4:
        wp = bp + 1e-3

    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    remap_gain = strength

    Y_final = (1-remap_gain)*Y + remap_gain*Y_remap

    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)

    lin *= gain

    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]

    blend = 0.65 * remap_gain
    Y_mix = (1-blend)*Y2 + blend*np.clip(Y_final*gain, 0, 2.5)

    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[...,None], 0, 4)

    out = filmic_tonemap(np.clip(lin, 0, 4))
    out = np.clip(out, 0, 1)
    out = linear_to_srgb(out)

    return np.clip(out, 0, 1)
# ============================================================
# Sidebar Controls — Full UI System v2025
# ============================================================

st.sidebar.header("1) Data Source (NewsAPI & Random)")

keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora, emotion)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
    placeholder="e.g., AI"
)

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate (Crystal Mode)")


# ============================================================
# Load Data (Random / NewsAPI / Default)
# ============================================================

df = pd.DataFrame()

# ❶ RANDOM MODE — Fully generative crystals
if random_btn:
    st.session_state["auto_seed"] = int(np.random.randint(0, 100000))

    rng = np.random.default_rng()
    N = 12

    texts = [f"Random crystal #{i+1}" for i in range(N)]
    emos = [f"crystal_{i+1}" for i in range(N)]

    # reset custom palette to random colors
    st.session_state["custom_palette"] = {}
    for i, emo in enumerate(emos, start=1):
        r = int(rng.integers(0,256))
        g = int(rng.integers(0,256))
        b = int(rng.integers(0,256))
        st.session_state["custom_palette"][emo] = (r,g,b)

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "timestamp": str(date.today()),
        "compound": 0,
        "pos":0, "neu":1, "neg":0,
        "source":"CrystalGen"
    })


# ❷ FETCH MODE
elif fetch_btn:
    st.session_state["auto_seed"] = int(np.random.randint(0, 100000))

    api_key = st.secrets.get("NEWS_API_KEY","")
    if not api_key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(api_key, keyword if keyword.strip() else "aurora")


# ❸ DEFAULT MODE
if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the night sky.",
        "Calm weather created a peaceful atmosphere.",
        "Anxiety rises in unstable market conditions.",
        "A rare moment of awe as the sky turns green.",
        "Hope grows as new discoveries emerge."
    ]})
    df["timestamp"] = str(date.today())


# ============================================================
# Sentiment + Emotion
# ============================================================

df["text"] = df["text"].fillna("")

if "emotion" not in df.columns:
    sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# ============================================================
# Sidebar • Emotion Filtering
# ============================================================

st.sidebar.header("2) Emotion Mapping")

cmp_min = st.sidebar.slider(
    "Compound Min",
    -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]), 0.01
)

cmp_max = st.sidebar.slider(
    "Compound Max",
    -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]), 0.01
)

# initialize state
init_palette_state()
base_palette = get_active_palette()


available_emotions = sorted(df["emotion"].unique().tolist())

# Label for UI
def _label_emotion(e: str):
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r,g,b = base_palette.get(e,(0,0,0))
    return f"{e} (Custom {r},{g},{b})"


# Auto-top3
auto_top3 = st.sidebar.checkbox(
    "Auto-select top 3 emotions",
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"])
)

top3 = []
if auto_top3:
    vc = df["emotion"].value_counts()
    top3 = vc.head(3).index.tolist()


# ============================================================
# SPECIAL CASE：CSV-only → Selected Emotions disabled
# ============================================================

if st.session_state.get("use_csv_palette", False):
    st.sidebar.write("Selected emotions are ignored in CSV-only mode.")
    selected_emotions = df["emotion"].unique().tolist()

else:
    option_labels = [_label_emotion(e) for e in available_emotions]
    default_labels = [_label_emotion(e) for e in (top3 if top3 else available_emotions)]

    selected_labels = st.sidebar.multiselect(
        "Selected Emotions",
        option_labels,
        default=default_labels
    )

    selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]


# Apply compound filtering + emotion filtering
df = df[
    (df["emotion"].isin(selected_emotions)) &
    (df["compound"] >= cmp_min) &
    (df["compound"] <= cmp_max)
]


# ============================================================
# CSV Auto-Rename (1A)
# ============================================================

df = apply_csv_palette_autorename(df)


# ============================================================
# Crystal Render Controls
# ============================================================

st.sidebar.subheader("3) Crystal Engine")

layer_count = st.sidebar.slider(
    "Layers",
    1, 30, 2
)

wobble_control = st.sidebar.slider(
    "Wobble (Crystal randomness)",
    0.00, 1.00,
    0.25, 0.01
)

seed_control = st.sidebar.slider(
    "Seed",
    0, 500,
    st.session_state.get("auto_seed", 25)
)

shapes_per_emotion = st.sidebar.slider(
    "Crystals per Emotion",
    1, 40,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"])
)

stroke_blur = st.sidebar.slider(
    "Crystal Softness (blur px)",
    0.0, 20.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"])
)

fill_alpha = st.sidebar.slider(
    "Crystal Alpha",
    40, 255,
    st.session_state.get("ribbon_alpha", DEFAULTS["ribbon_alpha"])
)

poly_min_size = st.sidebar.slider(
    "Min Crystal Size",
    20, 300,
    st.session_state.get("poly_min_size", DEFAULTS["poly_min_size"])
)

poly_max_size = st.sidebar.slider(
    "Max Crystal Size",
    60, 600,
    st.session_state.get("poly_max_size", DEFAULTS["poly_max_size"])
)


# ============================================================
# Background Color
# ============================================================

st.sidebar.subheader("4) Background")
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

bg_custom = st.sidebar.color_picker(
    "Choose Background Color",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"])
)
bg_rgb = _hex_to_rgb(bg_custom)


# ============================================================
# Cinematic Color (Normal mode only)
# ============================================================

st.sidebar.header("5) Cinematic Color (Disabled in CSV-only)")

if not st.session_state.get("use_csv_palette", False):

    exp = st.sidebar.slider("Exposure", -0.2, 1.8,
        st.session_state.get("exp", DEFAULTS["exp"]), 0.01
    )
    contrast = st.sidebar.slider("Contrast", 0.7, 1.8,
        st.session_state.get("contrast", DEFAULTS["contrast"])
    )
    saturation = st.sidebar.slider("Saturation", 0.7, 1.9,
        st.session_state.get("saturation", DEFAULTS["saturation"])
    )
    gamma_val = st.sidebar.slider("Gamma", 0.7, 1.4,
        st.session_state.get("gamma_val", DEFAULTS["gamma_val"])
    )
    roll = st.sidebar.slider("Highlight Roll-off", 0.0, 1.5,
        st.session_state.get("roll", DEFAULTS["roll"])
    )

    # White balance
    st.sidebar.subheader("White Balance")
    temp = st.sidebar.slider("Temperature", -1.0, 1.0,
        st.session_state.get("temp", DEFAULTS["temp"])
    )
    tint = st.sidebar.slider("Tint", -1.0, 1.0,
        st.session_state.get("tint", DEFAULTS["tint"])
    )

    # Split toning
    st.sidebar.subheader("Split Toning")
    sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", DEFAULTS["sh_r"]))
    sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", DEFAULTS["sh_g"]))
    sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", DEFAULTS["sh_b"]))

    hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", DEFAULTS["hi_r"]))
    hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", DEFAULTS["hi_g"]))
    hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", DEFAULTS["hi_b"]))

    tone_balance = st.sidebar.slider("Tone Balance", -1.0, 1.0,
        st.session_state.get("tone_balance", DEFAULTS["tone_balance"])
    )

    # Bloom & Vignette
    st.sidebar.subheader("Bloom & Vignette")
    bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
        st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"])
    )
    bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
        st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"])
    )
    vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
        st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"])
    )

    # Auto brightness
    st.sidebar.header("6) Auto Brightness")
    auto_bright = st.sidebar.checkbox(
        "Enable Auto Brightness",
        value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"])
    )
    target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70,
        st.session_state.get("target_mean", DEFAULTS["target_mean"])
    )
    abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0,
        st.session_state.get("abc_strength", DEFAULTS["abc_strength"])
    )
    abc_black = st.sidebar.slider("Black Point %", 0.00, 0.20,
        st.session_state.get("abc_black", DEFAULTS["abc_black"])
    )
    abc_white = st.sidebar.slider("White Point %", 0.80, 1.00,
        st.session_state.get("abc_white", DEFAULTS["abc_white"])
    )
    abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0,
        st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"])
    )

else:
    st.sidebar.write("Cinematic Color disabled (CSV palette only mode)")


# ============================================================
# CSV Palette Controls
# ============================================================

st.sidebar.header("7) Custom Palette (CSV / RGB)")

use_csv = st.sidebar.checkbox(
    "Use CSV palette only (disable defaults & PostFX)",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

# Import CSV
csv_file = st.sidebar.file_uploader("Import Palette CSV", type=["csv"])
if csv_file:
    import_palette_csv(csv_file)

# Export
if st.sidebar.button("Export Current Palette CSV"):
    buf = export_palette_csv(st.session_state.get("custom_palette", {}))
    st.sidebar.download_button(
        "Download CSV",
        data=buf,
        file_name="palette_export.csv",
        mime="text/csv"
    )


# ============================================================
# Reset
# ============================================================

st.sidebar.header("8) Output Control")
if st.sidebar.button("Reset All", type="primary"):
    reset_all()
# ============================================================
# PART 6 — Rendering + PostFX System (Stable v2025)
# ============================================================

left, right = st.columns([0.60, 0.40])

# ============================================================
# LEFT — Crystal Rendering
# ============================================================

with left:
    st.subheader("❄️ Crystal Mix Visualization")

    # -------------------------------
    # Determine active palette
    # -------------------------------
    if st.session_state.get("use_csv_palette", False):
        # STRICT MODE → only CSV colors
        working_palette = dict(st.session_state.get("custom_palette", {}))
    else:
        # Hybrid mode → DEFAULT + CUSTOM
        working_palette = get_active_palette()

    # -------------------------------
    # Render base crystal image
    # -------------------------------
    img = render_crystalmix(
        df=df,
        palette=working_palette,
        width=1500,
        height=850,
        seed=seed_control,
        shapes_per_emotion=shapes_per_emotion,
        min_size=poly_min_size,
        max_size=poly_max_size,
        fill_alpha=int(fill_alpha),
        blur_px=int(stroke_blur),
        bg_color=bg_rgb,
        wobble=wobble_control,
        layers=layer_count
    )

    # -------------------------------
    # Convert PIL → NumPy
    # -------------------------------
    arr = np.array(img).astype(np.float32) / 255.0


    # ============================================================
    # STRICT CSV-ONLY MODE — No PostFX Allowed
    # ============================================================

    if st.session_state.get("use_csv_palette", False):

        final_img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        buf = BytesIO()
        final_img.save(buf, format="PNG")
        buf.seek(0)

        # Show + Download
        st.image(final_img, use_column_width=True)
        st.download_button(
            "💾 Download PNG (True CSV Color — No PostFX)",
            data=buf,
            file_name="crystal_mix_truecolor.png",
            mime="image/png"
        )

        st.stop()   # 🔥 prevent PostFX from running


    # ============================================================
    # NORMAL MODE — Full Cinematic PostFX Pipeline
    # ============================================================

    # 1) Convert to linear HDR
    lin = srgb_to_linear(arr)

    # 2) Exposure
    lin = lin * (2.0 ** exp)

    # 3) White balance
    lin = apply_white_balance(lin, temp, tint)

    # 4) Highlight roll-off
    lin = highlight_rolloff(lin, roll)

    # 5) Back to display-referred sRGB
    arr = linear_to_srgb(np.clip(lin, 0, 4))

    # 6) Filmic tone map
    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)

    # 7) Local adjustments
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    # 8) Split toning
    arr = split_tone(
        arr,
        sh_rgb=(sh_r, sh_g, sh_b),
        hi_rgb=(hi_r, hi_g, hi_b),
        balance=tone_balance
    )

    # 9) Auto brightness
    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    # 10) Bloom + vignette
    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)

    # 11) Ensure colorfulness
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    # -------------------------------
    # Convert back to PIL & export
    # -------------------------------
    final_img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    st.image(final_img, use_column_width=True)
    st.download_button(
        "💾 Download PNG",
        data=buf,
        file_name="crystal_mix.png",
        mime="image/png"
    )
# ============================================================
# PART 7 — Right Panel: Data Table + Emotion Display
# ============================================================

with right:

    st.subheader("📊 Data & Emotion Mapping")

    # -------------------------------
    # Prepare formatted DataFrame
    # -------------------------------
    df2 = df.copy()

    # Label emotion with human-friendly names or CSV indicator
    def format_emo(e):
        # Case A: CSV palette-only → custom value
        if st.session_state.get("use_csv_palette", False):
            if e in st.session_state["custom_palette"]:
                r, g, b = st.session_state["custom_palette"][e]
                return f"{e} (CSV {r},{g},{b})"
            else:
                return f"{e} (CSV-Unknown)"
        
        # Case B: Hybrid → use default naming
        if e in COLOR_NAMES:
            return f"{e} ({COLOR_NAMES[e]})"
        else:
            r, g, b = base_palette.get(e, (0,0,0))
            return f"{e} (Custom {r},{g},{b})"

    df2["emotion_display"] = df2["emotion"].apply(format_emo)

    # Ordering of columns
    cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]
    if "timestamp" in df2.columns:
        cols.insert(1, "timestamp")
    if "source" in df2.columns:
        cols.insert(2, "source")

    # -------------------------------
    # Render table
    # -------------------------------
    st.dataframe(
        df2[cols],
        use_container_width=True,
        height=620
    )

    # -------------------------------
    # Palette Preview (optional)
    # -------------------------------
    st.divider()
    st.markdown("### 🎨 Active Palette (Preview)")

    # Determine active palette again
    if st.session_state.get("use_csv_palette", False):
        pal_to_show = dict(st.session_state.get("custom_palette", {}))
    else:
        pal_to_show = get_active_palette()

    # Build a small preview grid
    if pal_to_show:
        preview_df = pd.DataFrame([
            {
                "Emotion": k,
                "R": v[0],
                "G": v[1],
                "B": v[2],
                "Color": f"rgb({v[0]},{v[1]},{v[2]})"
            }
            for k, v in pal_to_show.items()
        ])

        st.dataframe(
            preview_df,
            use_container_width=True,
            height=min(500, 40*(len(preview_df)+1))
        )
    else:
        st.info("No custom colors loaded.")
