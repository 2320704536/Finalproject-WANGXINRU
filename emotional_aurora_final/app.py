# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Ribbon — Final", page_icon="🎐", layout="wide")
st.title("🎐 Emotional Ribbon — Final (Colorful Polygon Mix)")

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
# Emotion colors — 高饱和、区分度强
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
            except: continue
            pal[emo]=(r,g,b)
        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV.")
    except Exception as e:
        st.error(f"CSV import error: {e}")

def export_palette_csv(pal):
    buf = BytesIO()
    pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]).to_csv(buf, index=False)
    buf.seek(0); return buf

# =========================
# Color helpers（保持亮、避免黑线）
# =========================
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

# =========================
# 多边形生成/绘制
# =========================
def random_convex_polygon(center, rng, min_r=30, max_r=180, sides=5):
    cx, cy = center
    angles = np.sort(rng.random(sides) * 2*np.pi)
    radii = rng.uniform(min_r, max_r, size=sides)
    jitter = rng.normal(1.0, 0.08, size=sides)
    pts = []
    for a, r, j in zip(angles, radii, jitter):
        rr = max(5.0, r*j)
        x = cx + rr*np.cos(a)
        y = cy + rr*np.sin(a)
        pts.append((float(x), float(y)))
    return pts

def draw_polygon_soft(canvas_rgba, pts, color01, fill_alpha=200, blur_px=6, edge_width=0):
    W, H = canvas_rgba.size
    layer = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")
    col = (int(color01[0]*255), int(color01[1]*255), int(color01[2]*255), int(fill_alpha))
    d.polygon(pts, fill=col)
    if edge_width > 0:
        edge = (255,255,255,max(80, fill_alpha//2))
        d.line(pts + [pts[0]], fill=edge, width=edge_width, joint="curve")
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))
    canvas_rgba.alpha_composite(layer)

def safe_text_bbox(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            w, h = draw.textsize(text, font=font)
            return w, h
        except Exception:
            return 0, 0

def add_title(img_rgb, title, color_rgb=(255,255,255)):
    W, H = img_rgb.size
    rgba = img_rgb.convert("RGBA")
    overlay = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(overlay, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=int(H*0.06))
    except Exception:
        font = ImageFont.load_default()
    tw, th = safe_text_bbox(d, title, font)
    pad = int(H*0.02)
    x, y = pad, pad
    bg_rect = [x-10, y-6, x+tw+16, y+th+10]
    d.rectangle(bg_rect, fill=(0,0,0,140), outline=None)
    d.text((x,y), title, font=font, fill=(color_rgb[0], color_rgb[1], color_rgb[2], 255))
    rgba.alpha_composite(overlay)
    return rgba.convert("RGB")

# =========================
# 渲染：多边形合成（替代丝带）
# =========================
def render_polymix(
    df, palette, width=1500, height=850, seed=12345,
    shapes_per_emotion=10,
    min_size=60, max_size=220,
    shape_sides_min=4, shape_sides_max=8,
    fill_alpha=210, blur_px=6,
    bg_color=(0,0,0)
):
    rng = np.random.default_rng(seed)
    base = Image.new("RGBA", (width, height), (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0,0,0,0))

    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["joy","love","curiosity"]

    for emo in emotions:
        base_rgb = palette.get(emo, palette.get("mixed", (230,190,110)))
        base01 = vibrancy_boost(base_rgb, sat_boost=1.30, min_luma=0.40)
        size_bias = 1.0
        if emo in ("joy","love","pride","surprise"):
            size_bias = 1.2
        elif emo in ("calm","nostalgia","awe","trust"):
            size_bias = 0.95

        for _ in range(max(1, int(shapes_per_emotion))):
            cx = rng.uniform(0.05*width, 0.95*width)
            cy = rng.uniform(0.08*height, 0.92*height)
            smin = max(20, int(min_size*size_bias*rng.uniform(0.8,1.2)))
            smax = max(smin+5, int(max_size*size_bias*rng.uniform(0.8,1.3)))
            sides = int(rng.integers(shape_sides_min, shape_sides_max+1))
            pts = random_convex_polygon((cx,cy), rng, min_r=smin, max_r=smax, sides=sides)

            col01 = jitter_color(base01, rng, amount=0.07)
            local_alpha = int(np.clip(fill_alpha * rng.uniform(0.85, 1.05), 40, 255))
            local_blur  = max(0, int(blur_px * rng.uniform(0.7, 1.4)))
            edge_w = 0 if rng.random() < 0.6 else max(1, int(smin*0.02))

            draw_polygon_soft(canvas, pts, col01,
                fill_alpha=local_alpha, blur_px=local_blur, edge_width=edge_w)

    base.alpha_composite(canvas)
    return base.convert("RGB")


# =========================
# Cinematic Color（可选）
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

def apply_white_balance(img, temp, tint):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    r *= (1.0 + 0.10*temp)
    b *= (1.0 - 0.10*temp)
    g *= (1.0 + 0.08*tint)
    r *= (1.0 - 0.06*tint)
    b *= (1.0 - 0.02*tint)
    out = np.stack([r,g,b], axis=-1)
    return np.clip(out, 0, 1)

def adjust_contrast(img, c):
    return np.clip((img - 0.5)*c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = lum[...,None]
    return np.clip(lum + (img - lum)*s, 0, 1)

def gamma_correct(img, g):
    return np.clip(img ** (1.0/g), 0, 1)

def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5)
    threshold = 0.8
    mask = np.clip((img - threshold)/(1e-6 + 1.0 - threshold), 0, 1)
    out = img*(1 - mask) + (threshold + (img-threshold)/(1.0 + 4.0*t*mask))*mask
    return np.clip(out, 0, 1)

def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = (lum - lum.min())/(lum.max()-lum.min()+1e-6)
    sh = np.clip(1.0 - lum + 0.5*(1-balance), 0, 1)[...,None]
    hi = np.clip(lum + 0.5*(1+balance) - 0.5, 0, 1)[...,None]
    sh_col = np.array(sh_rgb).reshape(1,1,3)
    hi_col = np.array(hi_rgb).reshape(1,1,3)
    out = np.clip(img + sh*sh_col*0.25 + hi*hi_col*0.25, 0, 1)
    return out

def apply_bloom(img, radius=6.0, intensity=0.6):
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8), mode="RGB")
    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32)/255.0
        out = np.clip(img*(1-intensity) + b*intensity, 0, 1)
        return out
    return img

def apply_vignette(img, strength=0.20):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2)/(w/2); yy = (yy - h/2)/(h/2)
    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength*(r**1.5), 0.0, 1.0)
    return np.clip(img * mask[...,None], 0, 1)

def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)
    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img


def auto_brightness_compensation(img_arr, target_mean=0.50, strength=0.9,
                                 black_point_pct=0.05, white_point_pct=0.997,
                                 max_gain=2.6):
    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)
    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-6: wp = bp + 1e-3
    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    remap_gain = np.clip(strength, 0, 1)
    Y_final = (1-remap_gain)*Y + remap_gain*Y_remap
    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)
    lin *= gain
    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    blend = 0.65*remap_gain
    Y_mix = (1-blend)*Y2 + blend*np.clip(Y_final*gain, 0, 2.5)
    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[...,None], 0, 4)
    out = filmic_tonemap(np.clip(lin,0,4))
    out = np.clip(out, 0, 1)
    out = linear_to_srgb(out)
    return np.clip(out, 0, 1)

# =========================
# Defaults & Reset
# =========================
DEFAULTS = {
    "keyword": "",
    "ribbons_per_emotion": 10,
    "stroke_width": 5,
    "steps": 520,
    "step_len": 2.4,
    "curve_noise": 0.34,
    "stroke_blur": 6.0,
    "ribbon_alpha": 210,
    "bg_custom": "#000000",

    "poly_min_size": 70,
    "poly_max_size": 220,
    "poly_sides_min": 4,
    "poly_sides_max": 8,

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
}

def reset_all():
    st.session_state.clear()
    st.rerun()


# =========================
# 使用说明
# =========================
with st.expander("How to Use", expanded=False):
    st.markdown("""
1) 在 **Keyword** 里输入关键词，点击 **Fetch News**  
2) 或者点击 **Random Generate** 随机生成 8 条文本  
3) 系统自动映射情绪 → 生成多边形 → 电影级调色  
4) 下载 PNG
""")


# =========================
# Sidebar — Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora borealis, technology, science)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
    placeholder="e.g., AI"
)

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate")   # ⭐ 新增按钮


# =========================
# Random / Fetch / Fallback
# =========================
df = pd.DataFrame()

# === NEW: random generate ===
if random_btn:
    sample_emotions = [
        "joy","love","pride","hope","curiosity","calm","surprise",
        "nostalgia","sadness","anger","fear","awe","gratitude","trust"
    ]
    rng = np.random.default_rng()
    fake_texts = []
    for _ in range(8):
        emo = rng.choice(sample_emotions)
        sentence = f"A moment of {emo} fills the atmosphere with shifting colors."
        fake_texts.append(sentence)

    df = pd.DataFrame({
        "text": fake_texts,
        "timestamp": str(date.today()),
        "source": "RandomGen"
    })

# === fetch news ===
elif fetch_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "aurora")

# === fallback ===
if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the northern sky last night.",
        "Calm atmospheric conditions create a beautiful environment.",
        "Anxiety spreads among investors during unstable market conditions.",
        "A moment of awe as the sky shines with green light.",
        "Hope arises as scientific discoveries advance our understanding."
    ]})
    df["timestamp"]=str(date.today())


# =========================
# Emotion Mapping
# =========================
df["text"] = df["text"].fillna("")
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================
# Sidebar — Emotion Filtering
# =========================
st.sidebar.header("2) Emotion Mapping")
cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]), 0.01, key="cmp_min")
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]), 0.01, key="cmp_max")

init_palette_state()
base_palette = get_active_palette()

available_emotions = sorted(df["emotion"].unique().tolist())

def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = base_palette.get(e, (0, 0, 0))
        return f"{e} (Custom {r},{g},{b})"

auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions after fetch",
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]),
    key="auto_top3"
)

top3 = []
if auto_top3 and len(df):
    vc = df["emotion"].value_counts()
    top3 = vc.head(3).index.tolist()

options_labels = [_label_emotion(e) for e in ALL_EMOTIONS]
default_labels = [_label_emotion(e) for e in (top3 if top3 else available_emotions)]
selected_labels = st.sidebar.multiselect("Selected Emotions", options_labels, default=default_labels)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

df = df[(df["emotion"].isin(selected_emotions)) & (df["compound"]>=cmp_min) & (df["compound"]<=cmp_max)]

# =========================
# Sidebar — Ribbon Engine（沿用名；用于多边形参数）
# =========================
st.sidebar.header("3) Ribbon Engine")
ribbons_per_emotion = st.sidebar.slider("Shapes per Emotion", 1, 40,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"]), 1, key="ribbons_per_emotion")
stroke_width = st.sidebar.slider("Stroke Width (unused for polygons)", 1, 24,
    st.session_state.get("stroke_width", DEFAULTS["stroke_width"]), 1, key="stroke_width")
steps = st.sidebar.slider("Ribbon Length (unused)", 120, 1500,
    st.session_state.get("steps", DEFAULTS["steps"]), 10, key="steps")
step_len = st.sidebar.slider("Step Length (unused)", 0.5, 8.0,
    st.session_state.get("step_len", DEFAULTS["step_len"]), 0.1, key="step_len")
curve_noise = st.sidebar.slider("Curve Randomness (unused)", 0.00, 0.90,
    st.session_state.get("curve_noise", DEFAULTS["curve_noise"]), 0.01, key="curve_noise")
stroke_blur = st.sidebar.slider("Polygon Softness (blur px)", 0.0, 20.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"]), 0.5, key="stroke_blur")
ribbon_alpha = st.sidebar.slider("Polygon Alpha", 40, 255,
    st.session_state.get("ribbon_alpha", DEFAULTS["ribbon_alpha"]), 5, key="ribbon_alpha")

# 多边形大小/边数
st.sidebar.subheader("Polygon Size & Sides")
poly_min_size = st.sidebar.slider("Min Size (px)", 20, 300,
    st.session_state.get("poly_min_size", DEFAULTS["poly_min_size"]), 5, key="poly_min_size")
poly_max_size = st.sidebar.slider("Max Size (px)", 60, 600,
    st.session_state.get("poly_max_size", DEFAULTS["poly_max_size"]), 5, key="poly_max_size")
poly_sides_min = st.sidebar.slider("Min Sides", 3, 10,
    st.session_state.get("poly_sides_min", DEFAULTS["poly_sides_min"]), 1, key="poly_sides_min")
poly_sides_max = st.sidebar.slider("Max Sides", 4, 12,
    st.session_state.get("poly_sides_max", DEFAULTS["poly_sides_max"]), 1, key="poly_sides_max")

# 背景（仅自定义颜色）
st.sidebar.subheader("Background (Solid Color)")
def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0,2,4))
bg_custom = st.sidebar.color_picker("Pick custom color", value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"]), key="bg_custom")
bg_rgb = _hex_to_rgb(bg_custom)

# =========================
# Sidebar — Cinematic Color System
# =========================
st.sidebar.header("4) Cinematic Color System")
exp = st.sidebar.slider("Exposure (stops)", -0.2, 1.8, st.session_state.get("exp", DEFAULTS["exp"]), 0.01, key="exp")
contrast = st.sidebar.slider("Contrast", 0.70, 1.80, st.session_state.get("contrast", DEFAULTS["contrast"]), 0.01, key="contrast")
saturation = st.sidebar.slider("Saturation", 0.70, 1.90, st.session_state.get("saturation", DEFAULTS["saturation"]), 0.01, key="saturation")
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40, st.session_state.get("gamma_val", DEFAULTS["gamma_val"]), 0.01, key="gamma_val")
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50, st.session_state.get("roll", DEFAULTS["roll"]), 0.01, key="roll")

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature (Blue ↔ Red)", -1.0, 1.0, st.session_state.get("temp", DEFAULTS["temp"]), 0.01, key="temp")
tint = st.sidebar.slider("Tint (Green ↔ Magenta)", -1.0, 1.0, st.session_state.get("tint", DEFAULTS["tint"]), 0.01, key="tint")

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", DEFAULTS["sh_r"]), 0.01, key="sh_r")
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", DEFAULTS["sh_g"]), 0.01, key="sh_g")
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", DEFAULTS["sh_b"]), 0.01, key="sh_b")
hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", DEFAULTS["hi_r"]), 0.01, key="hi_r")
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", DEFAULTS["hi_g"]), 0.01, key="hi_g")
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", DEFAULTS["hi_b"]), 0.01, key="hi_b")
tone_balance = st.sidebar.slider("Tone Balance (Shadows ↔ Highlights)", -1.0, 1.0, st.session_state.get("tone_balance", DEFAULTS["tone_balance"]), 0.01, key="tone_balance")

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius (px)", 0.0, 20.0, st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]), 0.5, key="bloom_radius")
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0, st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]), 0.01, key="bloom_intensity")
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8, st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]), 0.01, key="vignette_strength")

# =========================
# Sidebar — Auto Brightness
# =========================
st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox("Enable Auto Brightness", value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"]), key="auto_bright")
target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70, st.session_state.get("target_mean", DEFAULTS["target_mean"]), 0.01, key="target_mean")
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, st.session_state.get("abc_strength", DEFAULTS["abc_strength"]), 0.05, key="abc_strength")
abc_black = st.sidebar.slider("Black Point %", 0.00, 0.20, st.session_state.get("abc_black", DEFAULTS["abc_black"]), 0.01, key="abc_black")
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00, st.session_state.get("abc_white", DEFAULTS["abc_white"]), 0.001, key="abc_white")
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0, st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"]), 0.05, key="abc_max_gain")

# =========================
# Sidebar — Palette (自定义/CSV)
# =========================
st.sidebar.header("6) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette only", value=st.session_state.get("use_csv_palette", False), key="use_csv_palette")

with st.sidebar.expander("Add Custom Emotion", False):
    col1,col2,col3,col4=st.columns([1.8,1,1,1])
    name=col1.text_input("Emotion", key="add_emo")
    r=col2.number_input("R",0,255,210, key="add_r")
    g=col3.number_input("G",0,255,190, key="add_g")
    b=col4.number_input("B",0,255,140, key="add_b")
    if st.button("Add", key="btn_add"):
        add_custom_emotion(name,r,g,b)
    show = st.session_state.get("custom_palette",{})
    if show:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in show.items()]),
                     use_container_width=True,height=150)

with st.sidebar.expander("Import / Export Palette CSV", False):
    up = st.file_uploader("Import CSV",type=["csv"], key="up_csv")
    if up is not None:
        import_palette_csv(up)
    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state.get("custom_palette", {}))
    if st.session_state.get("use_csv_palette", False):
        pal = dict(st.session_state.get("custom_palette", {}))
    if pal:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]),
                     use_container_width=True,height=160)
        dl = export_palette_csv(pal)
        st.download_button("Download CSV",data=dl,file_name="palette.csv",mime="text/csv", key="dl_csv")

# =========================
# Sidebar — Reset
# =========================
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All", type="primary"):
    reset_all()

# =========================
# DRAW
# =========================
left, right = st.columns([0.60,0.40])

with left:
    st.subheader("🎐 Polygon Mix (Colorful)")

    working_palette = get_active_palette()

    # 多边形合成渲染（复用“ribbons_per_emotion”等值）
    img = render_polymix(
        df=df, palette=working_palette,
        width=1500, height=850, seed=np.random.randint(0, 999999),
        shapes_per_emotion=ribbons_per_emotion,
        min_size=poly_min_size, max_size=poly_max_size,
        shape_sides_min=poly_sides_min, shape_sides_max=poly_sides_max,
        fill_alpha=int(ribbon_alpha), blur_px=int(stroke_blur),
        bg_color=bg_rgb
    )

    # 电影级后期
    arr = np.array(img).astype(np.float32)/255.0
    lin = srgb_to_linear(arr)
    lin = lin * (2.0 ** exp)
    lin = apply_white_balance(lin, temp, tint)
    lin = highlight_rolloff(lin, roll)
    arr = linear_to_srgb(np.clip(lin, 0, 4))
    arr = np.clip(filmic_tonemap(arr*1.20), 0, 1)
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)
    arr = split_tone(arr, (st.session_state.get("sh_r", DEFAULTS["sh_r"]),
                           st.session_state.get("sh_g", DEFAULTS["sh_g"]),
                           st.session_state.get("sh_b", DEFAULTS["sh_b"])),
                          (st.session_state.get("hi_r", DEFAULTS["hi_r"]),
                           st.session_state.get("hi_g", DEFAULTS["hi_g"]),
                           st.session_state.get("hi_b", DEFAULTS["hi_b"])),
                           st.session_state.get("tone_balance", DEFAULTS["tone_balance"]))

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
    buf=BytesIO(); final_img.save(buf, format="PNG"); buf.seek(0)
    st.image(buf,use_column_width=True)
    st.download_button("💾 Download PNG",data=buf,file_name="emotion_polymix.png",mime="image/png")

with right:
    st.subheader("📊 Data & Emotion")
    df2=df.copy()
    df2["emotion_display"]=df2["emotion"].apply(lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})")
    cols=["text","emotion_display","compound","pos","neu","neg"]
    if "timestamp" in df.columns: cols.insert(1,"timestamp")
    if "source" in df.columns: cols.insert(2,"source")
    st.dataframe(df2[cols],use_container_width=True,height=600)

st.markdown("---")
st.caption("© 2025 Emotional Ribbon — Colorful Polygon Mix Edition")
