# ============================================================
# Emotional Crystal — FINAL CSV EDITION (Part 1 / 6)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date, datetime, timezone, timedelta
import math
import plotly.express as px


# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Crystal — Final CSV", page_icon="❄️", layout="wide")
st.title("❄️ Emotional Crystal — Final (CSV Edition)")

# Instructions
with st.expander("📘 Instructions", expanded=False):
    st.markdown("""
### 📘 How to Use

This app transforms **news + emotions + colors** into a cinematic **crystal visualization** and a set of analytic views and article recommendations.

---

## 1️⃣ Data Source (NewsAPI & Random Mode)

### 📰 NewsAPI Mode
- Add your **NEWS_API_KEY** in Streamlit Secrets  
- Enter a keyword (e.g. `AI`, `finance`, `aurora`) in the sidebar  
- Click **Fetch News**  
- The app will:
  - retrieve related news articles
  - extract `source / author / title / description / url / publishedAt`
  - run sentiment analysis (`pos / neu / neg / compound`)
  - map each item to an **emotion label**
  - compute a **recommendation score** and normalize it to **0–100**

### 🎲 Random Crystal Mode
- Click **Random Generate (Crystal Mode)**  
- Creates abstract visuals using random text, random emotion labels, and random RGB colors (no real news involved).

---

## 2️⃣ Crystal Mix Visualization (Top Section)

At the top of the page you see the main **Crystal Mix Visualization**:

- Each text item (news or random) is mapped to an emotion  
- Each emotion corresponds to a color (from the default palette or your CSV palette)  
- Colors are rendered as multiple soft polygon “crystal fragments”  
- Layers of fragments are stacked to form the final image  

You can control in the sidebar:
- number of layers  
- crystals per emotion  
- softness / blur / alpha  
- min & max size of fragments  
- wobble (shape irregularity)  
- random seed  
- solid background color  

---

## 3️⃣ Emotion & Color Controls

### Emotion Mapping
- Sentiment scores are mapped to extended emotion labels (e.g. **joy, calm, awe, anxiety, pride, sadness, mixed**).
- In the **Emotion Mapping** section you can:
  - filter by compound score range  
  - auto-select the top-3 most frequent emotions  
  - manually choose which emotions to keep in the visualization  

### CSV Color Palette
- Import your own CSV palette with columns: `emotion, r, g, b`  
- Turn on **Use CSV palette only** to rely **only** on your custom colors  
- Manually add new emotion–RGB entries  
- Export the current working palette as a CSV file  

---

## 4️⃣ Cinematic Color Grading

After the crystal image is rendered, a film-style color grading pipeline is applied:

- **Exposure, Contrast, Saturation, Gamma**  
- **White Balance** (temperature & tint)  
- **Split Toning** for shadows and highlights  
- **Highlight roll-off** for softer bright areas  
- **Bloom** (glow) and **Vignette**  
- **Auto Brightness Compensation** to avoid overly dark or blown-out results  

These controls give the image a more cinematic and polished look.

---

## 5️⃣ Analysis & Recommendations (Below the Crystal Image)

Below the Crystal Mix Visualization, the app shows four sections in this order:

1. **⭐ Top Recommendations**  
   - Each article gets a recommendation score based on:  
     - recency (freshness time windows: <6h, <24h, <72h, <7d, older)  
     - title length bonus if between 40–100 characters  
     - keyword match in title or description  
   - Scores are normalized to **0–100**  
   - The top articles are displayed as glass-style cards with:  
     - title, summary, source, publish time, score, and a link

2. **📊 Data & Emotion Mapping**  
   - A table of all items currently used in the crystal, including:  
     - text  
     - timestamp / source (if available)  
     - emotion label + color name  
     - sentiment scores (compound / pos / neu / neg)

3. **🗞 Article Analysis → DataFrame**  
   - A structured view of parsed articles, showing:  
     - source, author, title, description, url, publishedAt  
     - raw recommendation score and normalized score (0–100)

4. **📊 Articles by Source**  
   - A Plotly bar chart counting how many articles come from each source  
   - Helps you see which media outlets dominate for the chosen keyword.

---

## 6️⃣ Export

- Use **💾 Download PNG** to save the final crystal image  
- Use **Download CSV** in the palette section to save the current color palette  

Enjoy exploring the intersection of **news, emotion, color, and cinematic visual design** ✨
""")


# =========================
# VADER (Sentiment)
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
# News API (FIXED VERSION)
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

        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return [], pd.DataFrame()   # ← FIX: always return 2 values

        articles = data.get("articles", [])

        return articles, pd.DataFrame()   # ← FIX: return list + placeholder df

    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return [], pd.DataFrame()   # ← FIX: return 2 objects always

# ================================================================
# Article → DataFrame + Recommendation Score System
# ================================================================

def parse_articles_to_df(raw_articles, keyword=""):
    """
    Convert NewsAPI raw article dicts into a clean DataFrame
    with scoring + normalization.
    """
    rows = []
    now = datetime.now(timezone.utc)

    for a in raw_articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        published_str = a.get("publishedAt")

        # Parse datetime
        try:
            published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except:
            published_dt = now

        # -------------------------------
        # Time freshness score
        # -------------------------------
        hours = (now - published_dt).total_seconds() / 3600
        if hours < 6:
            time_score = 5
        elif hours < 24:
            time_score = 4
        elif hours < 72:
            time_score = 3
        elif hours < 24*7:
            time_score = 2
        else:
            time_score = 1

        # -------------------------------
        # Title length score
        # -------------------------------
        title_score = 1.5 if 40 <= len(title) <= 100 else 0

        # -------------------------------
        # Keyword match score
        # -------------------------------
        combine_txt = f"{title} {desc}".lower()
        keyword_score = 2 if keyword.lower() in combine_txt and keyword else 0

        total_score = time_score + title_score + keyword_score

        rows.append({
            "source": (a.get("source") or {}).get("name", ""),
            "author": a.get("author"),
            "title": title,
            "description": desc,
            "url": a.get("url"),
            "publishedAt": published_dt,
            "score": total_score
        })

    df = pd.DataFrame(rows)

    # -------------------------------
    # Normalize score to 0–100
    # -------------------------------
    if not df.empty:
        min_s = df["score"].min()
        max_s = df["score"].max()
        if max_s != min_s:
            df["score_norm"] = (df["score"] - min_s) / (max_s - min_s) * 100
        else:
            df["score_norm"] = 50
    else:
        df["score_norm"] = 0

    return df


# =========================
# Default Emotion Colors (Used only when CSV-only = OFF)
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

COLOR_NAMES = {
    "joy":"Jupiter Gold","love":"Rose","pride":"Violet","hope":"Mint",
    "curiosity":"Azure","calm":"Indigo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber"
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())

# =========================
# Sentiment → Emotion
# =========================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    # Positive group
    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"

    # Mixed / curiosity / awe
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    # Neutral / calm / boredom
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"
    if neu > 0.75 and abs(comp) < 0.1: return "boredom"

    # Negative group
    if comp <= -0.65 and neg > 0.5: return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45: return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3: return "anxiety"
    if neg > 0.45 and pos < 0.1: return "disgust"

    return "neutral"

# =========================
# Palette State (CSV + custom)
# =========================
def init_palette_state():
    # whether CSV-only mode is enabled
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False

    # custom palette storage (from CSV or user addition)
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    """
    Returns working palette:
    - If CSV-only ON → ONLY use custom_palette
    - Else → DEFAULT_RGB + custom_palette override
    """
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))

    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged

def add_custom_emotion(name, r, g, b):
    """Allows adding new emotion colors manually or in CSV."""
    if not name:
        return
    st.session_state["custom_palette"][name.strip()] = (int(r), int(g), int(b))

# =========================
# Crystal Shape (❄️)
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
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((float(x), float(y)))

    pts.append(pts[0])  # close shape
    return pts

# =========================
# Soft Polygon Drawing (for crystals)
# =========================
def draw_polygon_soft(canvas_rgba, pts, color01,
                      fill_alpha=200, blur_px=6, edge_width=0):

    W, H = canvas_rgba.size
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")

    col = (
        int(color01[0]*255),
        int(color01[1]*255),
        int(color01[2]*255),
        int(fill_alpha)
    )

    d.polygon(pts, fill=col)

    # white edge highlight
    if edge_width > 0:
        edge = (255,255,255, max(80, fill_alpha//2))
        d.line(pts, fill=edge, width=edge_width, joint="curve")

    # softness
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

    canvas_rgba.alpha_composite(layer)
# =========================
# Color Helper Functions
# =========================
def _rgb01(rgb):
    """Convert (R,G,B) 0-255 to 0-1 float."""
    c = np.array(rgb, dtype=np.float32) / 255.0
    return np.clip(c, 0, 1)

def vibrancy_boost(rgb, sat_boost=1.28, min_luma=0.38):
    """Boost saturation while keeping minimum brightness for crystals."""
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]

    # Raise brightness if too dark (ensures crystals are visible)
    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)

    # Saturation expansion
    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    c = np.clip(lum + (c - lum)*sat_boost, 0, 1)
    return tuple(c)

def jitter_color(rgb01, rng, amount=0.06):
    """Slight color randomness for more organic fragments."""
    j = (rng.random(3) - 0.5) * 2 * amount
    c = np.clip(np.array(rgb01) + j, 0, 1)
    return tuple(c.tolist())

# =========================
# Crystal Mix Renderer (CORE)
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
    """
    df: DataFrame with 'emotion'
    palette: emotion → (R,G,B)
    If CSV-only is ON → palette contains ONLY CSV colors
    """

    rng = np.random.default_rng(seed)

    # Canvas setup
    base = Image.new("RGBA", (width, height), (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0,0,0,0))

    # emotions sorted by frequency
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        # Safe fallback: if DataFrame is empty, no default yellow
        # Just use any color from palette
        if palette:
            emotions = list(palette.keys())
        else:
            emotions = ["joy"]

    # Final: ensure emotions ONLY use available palette keys
    emotions = [e for e in emotions if e in palette]

    if not emotions:
        # final fallback if palette is empty
        emotions = ["joy"]
        palette = {"joy": (200,200,255)}

    # Multi-layer generation
    for _layer in range(layers):
        for emo in emotions:

            # 100% use palette color (CSV-only handled earlier)
            base_rgb = palette[emo]
            base01 = vibrancy_boost(base_rgb, sat_boost=1.30, min_luma=0.40)

            # Multiple fragments per emotion
            for _ in range(max(1, int(shapes_per_emotion))):

                # random center
                cx = rng.uniform(0.05*width, 0.95*width)
                cy = rng.uniform(0.08*height, 0.92*height)

                # random radius
                rr = int(rng.uniform(min_size, max_size))

                # crystal shape
                pts = crystal_shape(
                    center=(cx,cy),
                    r=rr,
                    wobble=wobble,
                    sides_min=5,
                    sides_max=10,
                    rng=rng
                )

                # slight jitter for variation
                col01 = jitter_color(base01, rng, amount=0.07)

                local_alpha = int(np.clip(fill_alpha * rng.uniform(0.85, 1.05), 40, 255))
                local_blur  = max(0, int(blur_px * rng.uniform(0.7, 1.4)))
                edge_w = 0 if rng.random() < 0.6 else max(1, int(rr*0.02))

                draw_polygon_soft(
                    canvas, pts, col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    # merge layers
    base.alpha_composite(canvas)
    return base.convert("RGB")
# =========================
# Cinematic Color System
# =========================

def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055)**2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, x * 12.92, 1.055*(x**(1/2.4)) - 0.055)

def filmic_tonemap(x):
    A = 0.22; B = 0.30; C = 0.10; D = 0.20; E = 0.01; F = 0.30
    return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F

def apply_white_balance(lin_img, temp, tint):
    """
    White balance:
    temp: -1 → blue, +1 → yellow
    tint: -1 → green, +1 → magenta
    """

    temp_strength = 0.6
    tint_strength = 0.5

    wb_temp = np.array([
        1.0 + temp * temp_strength,   # R
        1.0,                           # G
        1.0 - temp * temp_strength    # B
    ])

    wb_tint = np.array([
        1.0 + tint * tint_strength,   # R
        1.0 - tint * tint_strength,   # G
        1.0 + tint * tint_strength    # B
    ])

    wb = wb_temp * wb_tint
    out = lin_img * wb.reshape(1,1,3)
    return np.clip(out, 0, 4)

def adjust_contrast(img, c):
    return np.clip((img - 0.5) * c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = lum[..., None]
    return np.clip(lum + (img - lum) * s, 0, 1)

def gamma_correct(img, g):
    return np.clip(img ** (1.0/g), 0, 1)

def highlight_rolloff(img, roll):
    threshold = 0.8
    t = np.clip(roll, 0.0, 1.5)
    mask = np.clip((img - threshold) / (1e-6 + 1.0 - threshold), 0, 1)

    out = img*(1 - mask) + (threshold + (img - threshold)/(1 + 4*t*mask)) * mask
    return np.clip(out, 0, 1)

def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum_norm = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    sh_mask = np.clip(1.0 - lum_norm + 0.5*(1 - balance), 0, 1)[...,None]
    hi_mask = np.clip(lum_norm + 0.5*(1 + balance) - 0.5, 0, 1)[...,None]

    sh_col = np.array(sh_rgb).reshape(1,1,3)
    hi_col = np.array(hi_rgb).reshape(1,1,3)

    out = img + sh_mask*sh_col*0.25 + hi_mask*hi_col*0.25
    return np.clip(out, 0, 1)

def apply_bloom(img, radius=6.0, intensity=0.6):
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8), mode="RGB")

    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32)/255.0
        out = img*(1-intensity) + b*intensity
        return np.clip(out, 0, 1)

    return img

def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2) / (w/2)
    yy = (yy - h/2) / (h/2)
    r = np.sqrt(xx*xx + yy*yy)

    mask = np.clip(1 - strength*(r**1.5), 0.0, 1.0)
    return np.clip(img * mask[...,None], 0, 1)

def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)

    sat = (mx - mn) / (mx + 1e-6)
    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img

# =========================
# Auto Brightness Compensation (NO ERROR)
# =========================
def auto_brightness_compensation(
    img_arr,
    target_mean=0.50,
    strength=0.9,
    black_point_pct=0.05,
    white_point_pct=0.997,
    max_gain=2.6
):
    """
    Fully repaired auto brightness system.
    No NameError. 100% safe.
    """

    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)

    # luminance
    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]

    # Black/White point
    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    wp = max(wp, bp + 1e-3)

    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)

    remap_gain = np.clip(strength, 0, 1)
    Y_final = (1-remap_gain)*Y + remap_gain*Y_remap

    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)

    lin *= gain

    # mix back highlight info
    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    blend = 0.65*remap_gain
    Y_mix = (1-blend)*Y2 + blend*np.clip(Y_final*gain, 0, 2.5)

    ratio = (Y_mix+1e-6) / (Y2+1e-6)
    lin = np.clip(lin * ratio[...,None], 0, 4)

    # Filmic
    out = filmic_tonemap(np.clip(lin,0,4))
    out = np.clip(out, 0, 1)
    out = linear_to_srgb(out)
    return np.clip(out, 0, 1)
# =========================
# CSV Palette Import
# =========================
def import_palette_csv(file):
    try:
        dfc = pd.read_csv(file)

        expected = {"emotion","r","g","b"}
        lower_cols = {c.lower(): c for c in dfc.columns}

        if not expected.issubset(lower_cols.keys()):
            st.error("CSV must contain: emotion, r, g, b")
            return

        pal = {}
        for _, row in dfc.iterrows():
            emo = str(row[lower_cols["emotion"]]).strip()
            try:
                r = int(row[lower_cols["r"]])
                g = int(row[lower_cols["g"]])
                b = int(row[lower_cols["b"]])
                pal[emo] = (r,g,b)
            except:
                continue

        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV successfully.")

    except Exception as e:
        st.error(f"CSV import error: {e}")


# =========================
# CSV Palette Export
# =========================
def export_palette_csv(pal):
    buf = BytesIO()
    df_out = pd.DataFrame(
        [{"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in pal.items()]
    )
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# =========================
# Safe Text Bounding Box
# =========================
def safe_text_bbox(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        return w, h
    except:
        try:
            w, h = draw.textsize(text, font=font)
            return w, h
        except:
            return 0, 0


# =========================
# Title Overlay (Optional)
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
    pad = int(H * 0.02)
    x, y = pad, pad

    # Background bar
    bg_rect = [x-10, y-6, x + tw + 16, y + th + 10]
    d.rectangle(bg_rect, fill=(0,0,0,140))
    d.text((x,y), title, font=font,
           fill=(color_rgb[0], color_rgb[1], color_rgb[2], 255))

    rgba.alpha_composite(overlay)
    return rgba.convert("RGB")


# =========================
# Defaults & Reset
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

    "temp": 0.0,
    "tint": 0.0,

    "sh_r": 0.08,
    "sh_g": 0.06,
    "sh_b": 0.16,
    "hi_r": 0.10,
    "hi_g": 0.08,
    "hi_b": 0.06,
    "tone_balance": 0.0,

    "vignette_strength": 0.16,
    "bloom_radius": 7.0,
    "bloom_intensity": 0.40,

    "cmp_min": -1.0,
    "cmp_max": 1.0,
    "auto_top3": True,

    "bg_custom": "#000000"
}

def reset_all():
    st.session_state.clear()
    st.rerun()



# =========================================================
# Sidebar — Data Source (NewsAPI)
# =========================================================
st.sidebar.header("1) Data Source (NewsAPI)")

keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora, science)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
    placeholder="e.g., AI"
)

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate (Crystal Mode)")


# =========================================================
# Load Data (Random Mode / NewsAPI / Default Demo)
# =========================================================
df = pd.DataFrame()

# ❄️ RANDOM CRYSTAL MODE
if random_btn:
    st.session_state["auto_seed"] = int(np.random.randint(0, 100000))
    rng = np.random.default_rng()

    num_items = 12
    texts, emos = [], []

    # 清空 palette（CSV-only 模式会用 CSV）
    st.session_state["custom_palette"] = {}

    for i in range(num_items):
        texts.append(f"Random crystal fragment #{i+1}")
        emo = f"crystal_{i+1}"
        emos.append(emo)

        r = int(rng.integers(0, 256))
        g = int(rng.integers(0, 256))
        b = int(rng.integers(0, 256))
        st.session_state["custom_palette"][emo] = (r, g, b)

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "timestamp": str(date.today()),
        "compound": 0,
        "pos": 0, "neu": 1, "neg": 0,
        "source": "CrystalGen"
    })


# 🔍 FETCH NEWS MODE
elif fetch_btn:
    st.session_state["auto_seed"] = int(np.random.randint(0, 100000))
    key = st.secrets.get("NEWS_API_KEY", "")

    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        raw_articles, _ = fetch_news(key, keyword if keyword.strip() else "aurora")

        if raw_articles:
            df_articles = parse_articles_to_df(raw_articles, keyword)
            st.session_state["df_articles"] = df_articles

        # original text for emotion
        texts = [(a.get("title") or "") + ". " + (a.get("description") or "")
                 for a in raw_articles]

        df = pd.DataFrame({
            "text": texts,
            "timestamp": [a.get("publishedAt","")[:10] for a in raw_articles],
            "source": [(a.get("source") or {}).get("name","") for a in raw_articles]
        })



# DEFAULT DEMO DATA
if df.empty:
    df = pd.DataFrame({
        "text": [
            "A breathtaking aurora illuminated the sky.",
            "Calm conditions create peaceful feelings.",
            "Anxiety rises among investors.",
            "A moment of awe as the sky glows green.",
            "Hope increases after scientific discoveries."
        ]
    })
    df["timestamp"] = str(date.today())


# =========================================================
# Sentiment Analysis & Emotion Classification
# =========================================================
df["text"] = df["text"].fillna("")

if "emotion" not in df.columns:
    sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================================================
# Emotion Filtering Sidebar
# =========================================================
st.sidebar.header("2) Emotion Mapping")

cmp_min = st.sidebar.slider(
    "Compound Min", -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]), 0.01
)
cmp_max = st.sidebar.slider(
    "Compound Max", -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]), 0.01
)

init_palette_state()

base_palette = get_active_palette()    # 包含 CSV-only 或 默认 palette

available_emotions = sorted(df["emotion"].unique().tolist())


def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    # custom color
    if e in base_palette:
        r, g, b = base_palette[e]
        return f"{e} (Custom {r},{g},{b})"
    return e


auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions",
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"])
)

top3 = []
if auto_top3:
    vc = df["emotion"].value_counts()
    top3 = vc.head(3).index.tolist()


option_labels = [_label_emotion(e) for e in available_emotions]
default_labels = [_label_emotion(e) for e in (top3 if top3 else available_emotions)]

selected_labels = st.sidebar.multiselect(
    "Selected Emotions",
    option_labels,
    default=default_labels
)

selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

df = df[(df["emotion"].isin(selected_emotions))
        & (df["compound"] >= cmp_min)
        & (df["compound"] <= cmp_max)]


# =========================================================
# Auto Seed Control
# =========================================================
if "auto_seed" not in st.session_state:
    st.session_state["auto_seed"] = 25

st.sidebar.subheader("Crystal Layer Controls")

layer_count = st.sidebar.slider("Layers", 1, 30, 2)
wobble_control = st.sidebar.slider("Wobble", 0.0, 1.0, 0.25, 0.01)
seed_control = st.sidebar.slider("Seed", 0, 500, st.session_state.get("auto_seed", 25))


# =========================================================
# Crystal Engine Sidebar
# =========================================================
st.sidebar.header("3) Crystal Engine")

ribbons_per_emotion = st.sidebar.slider("Crystals per Emotion", 1, 40,
                                        st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"]))
stroke_blur = st.sidebar.slider("Crystal Softness", 0.0, 20.0,
                                st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"]))
ribbon_alpha = st.sidebar.slider("Crystal Alpha", 40, 255,
                                 st.session_state.get("ribbon_alpha", DEFAULTS["ribbon_alpha"]))

st.sidebar.subheader("Crystal Size")
poly_min_size = st.sidebar.slider("Min Size", 20, 300,
                                  st.session_state.get("poly_min_size", DEFAULTS["poly_min_size"]))
poly_max_size = st.sidebar.slider("Max Size", 60, 600,
                                  st.session_state.get("poly_max_size", DEFAULTS["poly_max_size"]))


# =========================================================
# Background Color Sidebar
# =========================================================
st.sidebar.subheader("Background (Solid Color)")

bg_custom = st.sidebar.color_picker(
    "Choose Background",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"])
)

def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0,2,4))

bg_rgb = _hex_to_rgb(bg_custom)


# =========================================================
# Cinematic Color Sidebar
# =========================================================
st.sidebar.header("4) Cinematic Color")

exp = st.sidebar.slider("Exposure", -0.2, 1.8, st.session_state.get("exp", DEFAULTS["exp"]), 0.01)
contrast = st.sidebar.slider("Contrast", 0.7, 1.8, st.session_state.get("contrast", DEFAULTS["contrast"]))
saturation = st.sidebar.slider("Saturation", 0.7, 1.9, st.session_state.get("saturation", DEFAULTS["saturation"]))
gamma_val = st.sidebar.slider("Gamma", 0.7, 1.4, st.session_state.get("gamma_val", DEFAULTS["gamma_val"]))
roll = st.sidebar.slider("Highlight Roll-off", 0.0, 1.5, st.session_state.get("roll", DEFAULTS["roll"]))

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature", -1.0, 1.0, st.session_state.get("temp", DEFAULTS["temp"]))
tint = st.sidebar.slider("Tint", -1.0, 1.0, st.session_state.get("tint", DEFAULTS["tint"]))

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", DEFAULTS["sh_r"]))
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", DEFAULTS["sh_g"]))
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", DEFAULTS["sh_b"]))

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", DEFAULTS["hi_r"]))
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", DEFAULTS["hi_g"]))
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", DEFAULTS["hi_b"]))
tone_balance = st.sidebar.slider("Tone Balance", -1.0, 1.0,
                                 st.session_state.get("tone_balance", DEFAULTS["tone_balance"]))


# =========================================================
# Bloom / Vignette Sidebar
# =========================================================
st.sidebar.subheader("Bloom & Vignette")

bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
                                 st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]))
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
                                    st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]))
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
                                      st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]))


# =========================================================
# Auto Brightness Sidebar
# =========================================================
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
# Sidebar — CSV Palette Section
# =========================
st.sidebar.header("6) Custom Palette (CSV)")

use_csv = st.sidebar.checkbox(
    "Use CSV palette only",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

with st.sidebar.expander("Add Custom Emotion (manual RGB)", False):
    col1, col2, col3, col4 = st.columns([1.6,1,1,1])
    emo_name = col1.text_input("Emotion Name")
    r = col2.number_input("R", 0, 255, 180)
    g = col3.number_input("G", 0, 255, 180)
    b = col4.number_input("B", 0, 255, 200)

    if st.button("Add Color"):
        add_custom_emotion(emo_name, r, g, b)

with st.sidebar.expander("Import / Export Palette CSV", False):
    up = st.file_uploader("Import CSV", type=["csv"])
    if up is not None:
        import_palette_csv(up)

    working = dict(DEFAULT_RGB)
    working.update(st.session_state.get("custom_palette", {}))

    if st.session_state.get("use_csv_palette", False):
        working = dict(st.session_state.get("custom_palette", {}))

    if working:
        df_pal = pd.DataFrame(
            [{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in working.items()]
        )
        st.dataframe(df_pal, use_container_width=True)

        dl = export_palette_csv(working)
        st.download_button("Download CSV", data=dl,
                           file_name="palette.csv", mime="text/csv")

# =========================================================
# Output Control (Reset)
# =========================================================
st.sidebar.header("7) Output Control")
if st.sidebar.button("Reset All", type="primary"):
    reset_all()


# =========================================================
# Page Layout — Left (image) / Right (data)
# =========================================================
left, right = st.columns([0.60, 0.40])


# =========================================================
# LEFT COLUMN — CRYSTAL IMAGE
# =========================================================
with left:
    st.subheader("❄️ Crystal Mix Visualization")

    # Working palette (CSV-only or merged)
    working_palette = get_active_palette()

    # Render crystal image
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

    # Convert to array
    arr = np.array(img).astype(np.float32) / 255.0

    # ===== POST PROCESSING =====

    lin = srgb_to_linear(arr)
    lin = lin * (2.0 ** exp)
    lin = apply_white_balance(lin, temp, tint)
    lin = highlight_rolloff(lin, roll)
    arr = linear_to_srgb(np.clip(lin, 0, 4))
    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    arr = split_tone(arr, sh_rgb=(sh_r, sh_g, sh_b),
                     hi_rgb=(hi_r, hi_g, hi_b),
                     balance=tone_balance)

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

    # Convert back
    final_img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="RGB")

    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    st.image(buf, use_column_width=True)

    st.download_button(
        "💾 Download PNG",
        data=buf,
        file_name="crystal_mix.png",
        mime="image/png"
    )

# =========================================================
# FULL-WIDTH SECTION BELOW CRYSTAL IMAGE
# =========================================================
st.markdown("---")

# ---------------------------------------------------------
# 1️⃣ Top Recommendations
# ---------------------------------------------------------
if "df_articles" in st.session_state:
    st.subheader("⭐ Top Recommendations")

    dfA = st.session_state["df_articles"]
    topN = dfA.sort_values("score_norm", ascending=False).head(5)

    for _, row in topN.iterrows():
        st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 16px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.15);
">
<h3 style="margin:0; color:white">{row.title}</h3>
<p style="color:#dddddd">{row.description}</p>
<b style="color:#aaaaaa">Source:</b> {row.source}  
<br>
<b style="color:#aaaaaa">Published:</b> {row.publishedAt.strftime('%Y-%m-%d %H:%M')}  
<br>
<b style="color:#aaaaaa">Score:</b> {row.score:.2f} (Norm: {row.score_norm:.1f})  
<br>
<a href="{row.url}" target="_blank">🔗 Read article</a>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2️⃣ Data & Emotion Mapping
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Data & Emotion Mapping")

df2 = df.copy()
df2["emotion_display"] = df2["emotion"].apply(
    lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})"
)

cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]
if "timestamp" in df.columns:
    cols.insert(1, "timestamp")
if "source" in df.columns:
    cols.insert(2, "source")

st.dataframe(df2[cols], use_container_width=True, height=400)


# ---------------------------------------------------------
# 3️⃣ Article Analysis → DataFrame
# ---------------------------------------------------------
if "df_articles" in st.session_state:
    st.markdown("---")
    st.subheader("🗞 Article Analysis → DataFrame")
    st.dataframe(dfA, use_container_width=True, height=400)


# ---------------------------------------------------------
# 4️⃣ Articles by Source
# ---------------------------------------------------------
if "df_articles" in st.session_state:
    st.markdown("---")
    st.subheader("📊 Articles by Source")

    df_count = dfA.groupby("source").size().reset_index(name="count")

    fig = px.bar(
        df_count, x="source", y="count",
        title="Articles by Source",
        color="count",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)
