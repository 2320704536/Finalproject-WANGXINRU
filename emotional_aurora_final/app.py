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
# App Setup
# =========================================================
st.set_page_config(page_title="Emotional Shine — Final", page_icon="✨", layout="wide")
st.title("✨ Emotional Shine — Constructivism Edition")

# =========================================================
# Load VADER
# =========================================================
@st.cache_resource
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()

# =========================================================
# NewsAPI Fetch
# =========================================================
def fetch_news(api_key, keyword="emotion", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword, "language": "en", "sortBy": "publishedAt",
        "pageSize": page_size, "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
        if data.get("status") != "ok":
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
    except:
        return pd.DataFrame()

# =========================================================
# Emotion Colors (High Saturation)
# =========================================================
EMO_RGB = {
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

# =========================================================
# Sentiment → Emotion
# =========================================================
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
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.5 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"
    return "neutral"

# =========================================================
# Constructivism Shine Renderer
# =========================================================
def draw_shine(image, top_emotion):
    W,H = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    base = EMO_RGB.get(top_emotion,(255,180,90))
    r,g,b = base

    # 1) radiating bars
    cx, cy = W//2, H//2
    for i in range(22):
        ang = np.deg2rad(i*(360/22))
        x2 = cx + np.cos(ang)*W*0.9
        y2 = cy + np.sin(ang)*H*0.9
        alpha = int(220 - i*6)
        draw.line([cx,cy,x2,y2], fill=(r,g,b,alpha), width=14)

    # 2) layered rectangles
    for k in range(6):
        pad = 40 + k*35
        alpha = 150 - k*18
        draw.rectangle(
            [pad,pad,W-pad,H-pad],
            outline=(255,255,255,alpha),
            width=4
        )

    # 3) big color block
    draw.rectangle([0,H*0.65,W,H],
                   fill=(r,g,b,60))

    # 4) Create highlight bloom
    blur = image.filter(ImageFilter.GaussianBlur(radius=18))
    image.alpha_composite(blur)

    return image


def add_title(image, text, top_color):
    W,H = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    r,g,b = top_color

    # Choose white/black text automatically
    bright = (r*0.299 + g*0.587 + b*0.114) > 140
    txt_color = (0,0,0) if bright else (255,255,255)

    ts = 70
    try:
        font = ImageFont.truetype("arial.ttf", ts)
    except:
        font = ImageFont.load_default()

    # ✅ FIX: textbbox instead of textsize
    bbox = draw.textbbox((0,0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(((W - tw)/2, H*0.42), text, fill=txt_color, font=font)

    return image



def render_shine(df, bg_rgb):
    W,H = 1500,850
    img = Image.new("RGBA",(W,H),(bg_rgb[0],bg_rgb[1],bg_rgb[2],255))

    # top emotion
    top = df["emotion"].value_counts().index[0]
    top_color = EMO_RGB.get(top,(255,180,90))

    # shine geometry
    img = draw_shine(img, top)

    # title
    txt = f"MAIN EMOTION · {top.upper()}"
    img = add_title(img, txt, top_color)

    return img

# =========================================================
# Sidebar Controls
# =========================================================
st.sidebar.header("1) Data Source")
keyword = st.sidebar.text_input("Keyword", "aurora")
btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if btn:
    key = st.secrets.get("NEWS_API_KEY","")
    df = fetch_news(key, keyword)

if df.empty:
    df = pd.DataFrame({
        "text":[
            "Aurora lights the sky with hope.",
            "Fear spreads in uncertain times.",
            "A moment of awe captured by photographers.",
            "Calm atmosphere brings peace.",
        ],
        "timestamp":[str(date.today())]*4
    })

df["text"] = df["text"].fillna("")
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True),sent_df.reset_index(drop=True)],axis=1)
df["emotion"] = df.apply(classify_emotion,axis=1)

# Color controls
st.sidebar.header("2) Style")
contrast = st.sidebar.slider("Contrast",0.5,2.0,1.2,0.01)
saturation = st.sidebar.slider("Saturation",0.5,2.0,1.3,0.01)
gamma_val = st.sidebar.slider("Gamma",0.6,1.4,1.0,0.01)
vig = st.sidebar.slider("Vignette",0.0,0.8,0.20,0.01)

bg_hex = st.sidebar.color_picker("Background", "#000000")
def hex2rgb(h):
    h=h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))
bg_rgb = hex2rgb(bg_hex)

# =========================================================
# Render
# =========================================================
st.subheader("✨ Shine Canvas Preview")

# Render Shine
img = render_shine(df, bg_rgb)

# Convert for processing
arr = np.array(img).astype(np.float32)/255.0
arr = (arr - 0.5)*contrast + 0.5
arr = np.clip(arr,0,1)
arr = arr**(1.0/gamma_val)
lum = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
arr = np.clip(lum[...,None] + (arr - lum[...,None])*saturation,0,1)

# vignette
H,W,_ = arr.shape
yy,xx = np.mgrid[0:H,0:W]
xx=(xx-W/2)/(W/2); yy=(yy-H/2)/(H/2)
mask=np.clip(1 - vig*(np.sqrt(xx*xx+yy*yy)**1.5),0,1)
arr = arr*mask[...,None]

final = Image.fromarray((arr*255).astype(np.uint8))

buf=BytesIO()
final.save(buf,format="PNG")
buf.seek(0)

st.image(final,use_column_width=True)
st.download_button("💾 Download PNG",buf,"shine_canvas.png","image/png")

# =========================================================
# Data Table
# =========================================================
st.subheader("📊 Extracted Data")
cols=["text","emotion","compound","pos","neu","neg"]
if "timestamp" in df.columns:
    cols.insert(1,"timestamp")
st.dataframe(df[cols],use_container_width=True,height=500)

st.markdown("---")
st.caption("© 2025 Emotional Shine — Constructivism Mode")
