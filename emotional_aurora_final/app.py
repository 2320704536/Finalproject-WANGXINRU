import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date


# ======================================================
# App Setup
# ======================================================
st.set_page_config(page_title="Emotional Shine — Polygon", page_icon="💎", layout="wide")
st.title("💎 Emotional Shine — Polygon Edition")


# ======================================================
# Load VADER
# ======================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


# ======================================================
# News API
# ======================================================
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
            st.warning("NewsAPI: " + str(data.get("message")))
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
        st.error(f"News fetch error: {e}")
        return pd.DataFrame()


# ======================================================
# Emotion Colors
# ======================================================
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
ALL_EMOTIONS = list(DEFAULT_RGB.keys())


# ======================================================
# Sentiment to Emotion
# ======================================================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0,"neu":1,"pos":0,"compound":0}
    return sia.polarity_scores(text)


def classify_emotion(row):
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
    if pos > 0.35 and neu > 0.4 and comp >= 0: return "trust"
    if pos > 0.30 and neu > 0.35 and abs(comp) <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    return "neutral"


# ======================================================
# Polygon Generator (Shine Style)
# ======================================================
def jitter_color(rgb, rng, amt=40):
    r,g,b = rgb
    return (
        int(np.clip(r + rng.integers(-amt, amt), 0,255)),
        int(np.clip(g + rng.integers(-amt, amt), 0,255)),
        int(np.clip(b + rng.integers(-amt, amt), 0,255)),
    )


def random_polygon(center, size, rng, irregularity=0.7, num=3):
    cx, cy = center
    pts = []
    for _ in range(num):
        angle = rng.random()*2*np.pi
        r = size * (0.5 + rng.random()*irregularity)
        x = cx + r*np.cos(angle)
        y = cy + r*np.sin(angle)
        pts.append((x,y))
    return pts


def render_shine_polygons(df, palette, bg_rgb, W=1500, H=900,
                          min_frag=180, max_frag=260, seed=1234):

    rng = np.random.default_rng(seed)

    img = Image.new("RGBA", (W,H), (bg_rgb[0],bg_rgb[1],bg_rgb[2],255))
    d = ImageDraw.Draw(img, "RGBA")

    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["hope","calm","awe"]

    total_frag = rng.integers(min_frag, max_frag)

    for _ in range(total_frag):
        emo = rng.choice(emotions)
        base = palette.get(emo, (200,200,200))
        col = jitter_color(base, rng)

        cx = rng.uniform(0,W)
        cy = rng.uniform(0,H)

        size = rng.uniform(40,200)
        num = rng.integers(3,8)

        pts = random_polygon((cx,cy), size, rng, irregularity=0.8, num=num)

        alpha = rng.integers(120,220)

        d.polygon(pts, fill=(col[0],col[1],col[2],alpha))

    # =============================
    # Add Title (Main Emotion)
    # =============================
    main_emo = df["emotion"].value_counts().idxmax()
    text_color = palette.get(main_emo, (250,240,200))

    draw2 = ImageDraw.Draw(img, "RGBA")
    txt = main_emo.capitalize()
    tw, th = draw2.textlength(txt), 56
    draw2.text((W-280, H-120), txt,
               fill=(text_color[0],text_color[1],text_color[2],255))

    return img.convert("RGB")


# ======================================================
# Sidebar — Data Source
# ======================================================
st.sidebar.header("1) Data Source")

keyword = st.sidebar.text_input("Keyword", "")
fetch_btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if key:
        df = fetch_news(key, keyword if keyword.strip() else "emotion")

if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the sky.",
        "Calm atmosphere brings comfort.",
        "Fear spreads during instability.",
        "Hope arises in scientific progress.",
        "Awe as colors shine in the night."
    ]})
    df["timestamp"] = str(date.today())

df["text"] = df["text"].fillna("")
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
df["emotion"] = df.apply(classify_emotion, axis=1)


# ======================================================
# Sidebar — Emotion Filter
# ======================================================
st.sidebar.header("2) Emotion Filter")

cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0, -1.0, 0.01)
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0, 1.0, 0.01)

df = df[(df["compound"]>=cmp_min) & (df["compound"]<=cmp_max)]


# ======================================================
# Sidebar — Background Color
# ======================================================
st.sidebar.header("3) Background")
bg_hex = st.sidebar.color_picker("Background", "#000000")
bg_rgb = tuple(int(bg_hex.lstrip("#")[i:i+2], 16) for i in (0,2,4))


# ======================================================
# Sidebar — Reset
# ======================================================
def reset_all():
    st.session_state.clear()
    st.rerun()

if st.sidebar.button("Reset All"):
    reset_all()


# ======================================================
# DRAW
# ======================================================
left, right = st.columns([0.60,0.40])

with left:
    st.subheader("💎 Shine Polygon Renderer")

    img = render_shine_polygons(
        df=df,
        palette=DEFAULT_RGB,
        bg_rgb=bg_rgb,
        seed=np.random.randint(0,999999)
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    st.image(buf, use_column_width=True)
    st.download_button("🔽 Download PNG", data=buf,
                       file_name="shine_polygon.png", mime="image/png")


with right:
    st.subheader("📊 Data / Emotion")
    st.dataframe(df, use_container_width=True, height=600)


st.markdown("---")
st.caption("© 2025 Emotional Shine — Polygon Style")
