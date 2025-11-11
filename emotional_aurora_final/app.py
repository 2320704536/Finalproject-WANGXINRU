import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# =========================
# App setup
# =========================
st.set_page_config(page_title="Geometric Emotion Poster", page_icon="🎨", layout="wide")
st.title("🎨 Geometric Emotion Poster — Final Edition")

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

# =========================
# Emotion colors
# =========================
DEFAULT_RGB = {
    "joy":        (255,200,60),    # 金黄
    "love":       (255,95,150),    # 玫红
    "pride":      (190,100,255),   # 电紫
    "hope":       (60,235,190),    # 明亮薄荷
    "curiosity":  (50,190,255),    # 天蓝
    "calm":       (70,135,255),    # 靛青
    "surprise":   (255,160,70),    # 杏橙
    "neutral":    (190,190,200),   # 灰蓝
    "sadness":    (80,120,230),    # 海蓝
    "anger":      (245,60,60),     # 红
    "fear":       (150,70,200),    # 紫
    "disgust":    (150,200,60),    # 黄绿
    "anxiety":    (255,200,60),    # 金色
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
    "curiosity":"Azure","calm":"Indigo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber"
}

# =========================
# Sentiment→Emotion
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
# Palette state
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))
    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged

def add_custom_emotion(name, r, g, b):
    if not name: return
    st.session_state["custom_palette"][name.strip()] = (int(r),int(g),int(b))

def import_palette_csv(file):
    try:
        dfc = pd.read_csv(file)
        need = {"emotion","r","g","b"}
        cols = {c.lower():c for c in dfc.columns}
        if not need.issubset(cols.keys()):
            st.error("CSV must include emotion,r,g,b columns"); return
        pal = {}
        for _, row in dfc.iterrows():
            emo = str(row[cols["emotion"]]).strip()
            try:
                r=int(row[cols["r"]]); g=int(row[cols["g"]]); b=int(row[cols["b"]])
            except:
                continue
            pal[emo]=(r,g,b)
        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV.")
    except Exception as e:
        st.error(f"CSV import error: {e}")

def export_palette_csv(pal):
    buf = BytesIO()
    pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]).to_csv(buf, index=False)
    buf.seek(0)
    return buf

# =========================
# UI Instruction
# =========================
with st.expander("How to Use", expanded=False):
    st.markdown("""
**Generate geometric emotion posters from real news text.**

✅ Enter a keyword (examples: AI, aurora, New York, Tesla)  
✅ System auto-detects emotions and auto-selects Top-3  
✅ Poster uses large abstract geometric shards  
✅ Customize color grading, brightness, palette  
✅ Download high-resolution PNG
""")
# =========================
# === Geometric Burst Renderer (核心新风格) ===
# =========================

def draw_polygon(canvas, pts, color, alpha=255):
    """在 RGBA canvas 上绘制一个硬边多边形"""
    layer = Image.new("RGBA", canvas.size, (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")
    col = (color[0], color[1], color[2], alpha)
    d.polygon(pts, fill=col)
    canvas.alpha_composite(layer)


def random_polygon(cx, cy, base_size, rng, angle_offset=0):
    """生成随机三角形 / 四边形碎片"""
    pts = []
    sides = rng.integers(3, 6)  # 3~5 边
    for i in range(sides):
        ang = angle_offset + (i / sides) * np.pi * 2 + rng.uniform(-0.25, 0.25)
        radius = base_size * rng.uniform(0.6, 1.4)
        x = cx + np.cos(ang) * radius
        y = cy + np.sin(ang) * radius
        pts.append((x, y))
    return pts


def render_geometric_burst(
    df,
    palette,
    width=1500,
    height=900,
    seed=123,
    bg_color=(0,0,0),
    burst_layers=28,
    base_size=180
):
    rng = np.random.default_rng(seed)

    # 背景
    bg = Image.new("RGBA", (width,height), (bg_color[0], bg_color[1], bg_color[2],255))
    canvas = Image.new("RGBA", (width,height), (0,0,0,0))

    # 使用出现频率排序的情绪
    emos = df["emotion"].value_counts().index.tolist()
    if not emos:
        emos = ["hope","calm","awe"]

    # 颜色池
    emo_colors = []
    for emo in emos:
        if emo in palette:
            emo_colors.append(palette[emo])
    if not emo_colors:
        emo_colors = [(200,200,200)]

    center_x = width * rng.uniform(0.40, 0.55)
    center_y = height * rng.uniform(0.40, 0.55)

    # 大爆炸层
    for i in range(burst_layers):

        # 每层选择一种情绪颜色
        col = emo_colors[i % len(emo_colors)]
        r,g,b = col

        # 亮度增强（使图和 Shine 风格一样亮丽）
        boost = rng.uniform(1.05, 1.30)
        r = min(int(r * boost), 255)
        g = min(int(g * boost), 255)
        b = min(int(b * boost), 255)

        # 每层生成若干碎片
        num_frag = rng.integers(3, 8)
        for _ in range(num_frag):

            # 中心偏移
            cx = center_x + rng.uniform(-80, 80)
            cy = center_y + rng.uniform(-80, 80)

            size = base_size * rng.uniform(0.5, 2.2)
            angle = rng.uniform(0, np.pi*2)

            pts = random_polygon(cx, cy, size, rng, angle_offset=angle)

            draw_polygon(canvas, pts, (r,g,b), alpha=rng.integers(190, 255))

    # 组合输出
    bg.alpha_composite(canvas)
    return bg.convert("RGB")
