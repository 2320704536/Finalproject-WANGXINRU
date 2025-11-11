import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date


# =========================================================
# App setup
# =========================================================
st.set_page_config(page_title="Emotional Shine — Final", page_icon="✨", layout="wide")
st.title("✨ Emotional Shine — Final Edition (Constructivism Style)")


# =========================================================
# Load VADER
# =========================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


# =========================================================
# Fetch News
# =========================================================
def fetch_news(api_key, keyword="emotion", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()

        rows = []
        for a in data.get("articles", []):
            text = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": text.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })

        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()


# =========================================================
# Emotion Colors (for Shine style)
# =========================================================
DEFAULT_RGB = {
    "joy": (255, 200, 70),
    "love": (255, 110, 160),
    "pride": (200, 110, 255),
    "hope": (95, 255, 205),
    "curiosity": (80, 210, 255),
    "calm": (70, 135, 255),
    "surprise": (255, 160, 90),
    "neutral": (200, 200, 205),
    "sadness": (90, 130, 230),
    "anger": (255, 75, 75),
    "fear": (160, 80, 225),
    "disgust": (160, 205, 70),
    "anxiety": (255, 215, 80),
    "boredom": (130, 130, 150),
    "nostalgia": (255, 225, 180),
    "gratitude": (120, 240, 230),
    "awe": (160, 245, 255),
    "trust": (80, 210, 170),
    "confusion": (255, 150, 190),
    "mixed": (230, 200, 130),
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy": "Joy",
    "love": "Love",
    "pride": "Pride",
    "hope": "Hope",
    "curiosity": "Curiosity",
    "calm": "Calm",
    "surprise": "Surprise",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "anger": "Anger",
    "fear": "Fear",
    "disgust": "Disgust",
    "anxiety": "Anxiety",
    "boredom": "Boredom",
    "nostalgia": "Nostalgia",
    "gratitude": "Gratitude",
    "awe": "Awe",
    "trust": "Trust",
    "confusion": "Confusion",
    "mixed": "Mixed"
}


# =========================================================
# Sentiment → Emotion
# =========================================================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return sia.polarity_scores(text)


def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp >= 0.70 and pos > 0.50: return "joy"
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
    if pos > 0.35 and neu > 0.4 and 0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"
    return "neutral"


# =========================================================
# Shine Engine — Geometry-based Shapes
# =========================================================
def draw_shine_shapes(img, df, palette, seed=12345):
    """生成 Constructivism风格的几何叠加图 + 自动主情绪标题"""
    rng = np.random.default_rng(seed)
    draw = ImageDraw.Draw(img, "RGBA")

    # Main emotion (Top-1)
    vc = df["emotion"].value_counts()
    main_emotion = vc.index[0] if len(vc) else "neutral"

    # shape colors
    emo_rgb = palette.get(main_emotion, (255, 200, 60))
    r, g, b = emo_rgb

    W, H = img.size

    # ============= Geometric Layers =============
    for i in range(14):
        cx = rng.uniform(0.15, 0.85) * W
        cy = rng.uniform(0.15, 0.85) * H
        max_r = rng.uniform(0.10, 0.45) * min(W, H)
        rr = int(max_r)
        alpha = int(rng.uniform(30, 95))

        # random shape choice
        shape_type = rng.choice(["circle", "triangle", "rect"])

        if shape_type == "circle":
            draw.ellipse(
                (cx - rr, cy - rr, cx + rr, cy + rr),
                fill=(r, g, b, alpha)
            )

        elif shape_type == "triangle":
            p1 = (cx, cy - rr)
            p2 = (cx - rr, cy + rr)
            p3 = (cx + rr, cy + rr)
            draw.polygon([p1, p2, p3], fill=(r, g, b, alpha))

        else:  # rect
            draw.rectangle(
                (cx - rr, cy - rr, cx + rr, cy + rr),
                fill=(r, g, b, alpha)
            )

    # ============= Add Shine Emotion Title =============
    title = COLOR_NAMES.get(main_emotion, main_emotion).upper()

    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", 72)
    except:
        font = ImageFont.load_default()

    tx = W - 40
    ty = H - 40

    tw, th = draw.textsize(title, font=font)
    tx -= tw
    ty -= th

    draw.text((tx, ty), title, fill=(255, 230, 120, 255), font=font)

    return img
# =========================================================
# Filmic + Color Processing
# =========================================================
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308, x*12.92, 1.055*(x**(1/2.4)) - 0.055)


def filmic_tonemap(x):
    A = 0.22; B = 0.30; C = 0.10; D = 0.20; E = 0.01; F = 0.30
    return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F


def adjust_contrast(img, c):
    return np.clip((img - 0.5)*c + 0.5, 0, 1)


def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = lum[...,None]
    return np.clip(lum + (img - lum)*s, 0, 1)


def gamma_correct(img, g):
    return np.clip(img ** (1.0/g), 0, 1)


def apply_vignette(img, strength=0.25):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2)/(w/2)
    yy = (yy - h/2)/(h/2)
    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength*(r**1.5), 0, 1)
    return img * mask[...,None]


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Settings")

contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.2, 0.01)
saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.3, 0.01)
gamma_val = st.sidebar.slider("Gamma", 0.6, 1.4, 1.0, 0.01)
vig = st.sidebar.slider("Vignette", 0.0, 0.8, 0.20, 0.01)

bg_color = st.sidebar.color_picker("Background Color", "#000000")
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))
bg_rgb = _hex_to_rgb(bg_color)


# =========================================================
# Data Processing
# =========================================================
st.sidebar.header("News Input")

keyword = st.sidebar.text_input("Keyword", "aurora")
fetch_btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(key, keyword)

# fallback sample
if df.empty:
    df = pd.DataFrame({
        "text":[
            "A breathtaking aurora illuminated the sky last night.",
            "Hope rises as researchers make progress.",
            "Fear spreads due to unexpected events.",
            "Moments of awe captured by photographers.",
        ],
        "timestamp":[str(date.today())]*4
    })

df["text"] = df["text"].fillna("")
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================================================
# Generate image
# =========================================================
st.subheader("✨ Shine Canvas Preview")

# Create base image (dark)
W, H = 1500, 850
img = Image.new("RGBA", (W, H), (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))

# Shine shapes + Title
img = draw_shine_shapes(img, df, DEFAULT_RGB, seed=np.random.randint(0, 999999))

# Convert to array
arr = np.array(img).astype(np.float32) / 255.0

# Filmic process
arr = adjust_contrast(arr, contrast)
arr = adjust_saturation(arr, saturation)
arr = gamma_correct(arr, gamma_val)
arr = apply_vignette(arr, vig)

final_img = Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8))

# display
buf = BytesIO()
final_img.save(buf, format="PNG")
buf.seek(0)

st.image(final_img, use_column_width=True)
st.download_button("💾 Download PNG", data=buf, file_name="shine_canvas.png", mime="image/png")


# =========================================================
# Table
# =========================================================
st.subheader("📊 Extracted Data")

cols = ["text", "emotion", "compound", "pos", "neu", "neg"]
if "timestamp" in df.columns:
    cols.insert(1, "timestamp")

st.dataframe(df[cols], use_container_width=True)

st.markdown("---")
st.caption("© 2025 Emotional Shine — Constructivism Edition")
