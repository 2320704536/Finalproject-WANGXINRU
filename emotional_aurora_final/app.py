import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Ribbon Flow — Wang Xinru — Final Project", page_icon="🌊", layout="wide")
st.title("🌊 Emotional Ribbon Flow — Wang Xinru — Final Project")

# =========================
# Instructions
# =========================
with st.expander("Instructions", expanded=False):
    st.markdown(
        """
**How to Use**

This project turns live news emotions into flowing ribbon visuals, then applies a cinematic color system with Auto Brightness Compensation so images are always bright and vivid.

1) Data → NewsAPI only. Enter a keyword and fetch.
2) Emotion Mapping → VADER to curated emotions; filter by compound range and by emotion list.
3) Ribbon Flow Engine → control bands per emotion, curve smoothness, ribbon width, blur, and flow randomness.
4) Cinematic Color System → exposure/contrast/gamma/saturation, white balance, split-toning, bloom, vignette, Auto Brightness.
5) Palette → one fixed color per emotion; add custom RGB; CSV import/export.
6) Download → save PNG.
"""
    )

# =========================
# Load VADER Sentiment
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
# News API Fetch
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
# Default planet-like emotion colors (base hues only; grading will add richness)
# =========================
DEFAULT_RGB = {
    "joy": (230,200,110),"love":(235,180,175),"pride":(200,170,210),"hope":(160,220,200),
    "curiosity":(175,210,200),"calm":(140,180,230),"surprise":(240,190,150),"neutral":(180,180,185),
    "sadness":(100,130,180),"anger":(180,80,70),"fear":(130,110,160),"disgust":(130,140,110),
    "anxiety":(210,190,140),"boredom":(120,120,130),"nostalgia":(235,220,190),"gratitude":(175,220,220),
    "awe":(190,230,240),"trust":(100,170,160),"confusion":(210,170,175),"mixed":(210,190,140),
}
ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy": "Warm Jupiter Gold","love": "Venus Rose","pride": "Saturn Violet","hope": "Uranus Mint",
    "curiosity": "Soft Turquoise","calm": "Neptune Blue","surprise": "Dawn Peach","neutral": "Lunar Gray",
    "sadness": "Deep Ocean Blue","anger": "Mars Red","fear": "Shadow Purple","disgust": "Olive Gray",
    "anxiety": "Desert Sand","boredom": "Slate Gray","nostalgia": "Pale Cream","gratitude": "Soft Cyan",
    "awe": "Ice Blue","trust": "Sea Teal","confusion": "Dust Pink","mixed": "Pale Gold",
}

# =========================
# Sentiment & Emotion mapping
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
    if st.session_state["use_csv_palette"]:
        return dict(st.session_state["custom_palette"])
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
# Color math utilities
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

def apply_vignette(img, strength=0.25):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2)/(w/2); yy = (yy - h/2)/(h/2)
    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength*(r**1.5), 0.0, 1.0)
    return np.clip(img * mask[...,None], 0, 1)

def ensure_colorfulness(img, min_sat=0.18, boost=1.25):
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)
    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img

# =========================
# Cinematic Palettes (global multipliers)
# =========================
CINEMATIC_PRESETS = {
    "Planetary (Soft)": {"mult": (1.00, 1.00, 1.00), "sat": 1.00, "temp": 0.00, "tint": 0.00},
    "Cinematic Cool":  {"mult": (0.95, 1.02, 1.08), "sat": 1.05, "temp": -0.20, "tint": 0.02},
    "Cinematic Warm":  {"mult": (1.08, 1.02, 0.95), "sat": 1.05, "temp": 0.20,  "tint": -0.02},
    "Neon Arctic":     {"mult": (0.90, 1.05, 1.15), "sat": 1.22, "temp": -0.30, "tint": 0.05},
    "Sunset Storm":    {"mult": (1.15, 1.03, 0.92), "sat": 1.20, "temp": 0.25,  "tint": 0.06},
    "Pastel Dream":    {"mult": (1.03, 1.03, 1.03), "sat": 0.92, "temp": 0.05,  "tint": 0.05},
    "Deep Space":      {"mult": (0.95, 0.98, 1.05), "sat": 0.96, "temp": -0.10, "tint": 0.00},
}

def apply_palette_preset(base_palette: dict, preset_name: str):
    p = CINEMATIC_PRESETS.get(preset_name, CINEMATIC_PRESETS["Planetary (Soft)"])
    mult = np.array(p["mult"])
    sat = p["sat"]
    out = {}
    for k, rgb in base_palette.items():
        col = np.array(rgb)/255.0
        col = np.clip(col * mult, 0, 1)
        col = adjust_saturation(col.reshape(1,1,3), sat)[0,0,:]
        out[k] = tuple((col*255).astype(int).tolist())
    return out, p["temp"], p["tint"]

def jitter_emotion_color(rgb, emo_key, amount=0.08):
    rng = np.random.default_rng(abs(hash(emo_key)) % (2**32))
    jitter = (rng.random(3)-0.5)*2*amount
    col = np.clip(np.array(rgb)/255.0 + jitter, 0, 1)
    return tuple((col*255).astype(int).tolist())

# =========================
# Auto Brightness Compensation
# =========================
def auto_brightness_compensation(img_arr, target_mean=0.46, strength=0.85,
                                 black_point_pct=0.06, white_point_pct=0.995,
                                 max_gain=2.4):
    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)
    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-6:
        wp = bp + 1e-3
    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    remap_gain = np.clip(strength, 0, 1)
    Y_final = (1-remap_gain)*Y + remap_gain*Y_remap
    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)
    lin *= gain
    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    blend = 0.6*remap_gain
    Y_mix = (1-blend)*Y2 + blend*np.clip(Y_final*gain, 0, 2.5)
    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[...,None], 0, 6)
    out = filmic_tonemap(np.clip(lin,0,6))
    out = np.clip(out, 0, 1)
    out = linear_to_srgb(out)
    return np.clip(out, 0, 1)

# =========================
# Ribbon Flow Engine
# =========================
def mix_colors(colors):
    if not colors:
        return np.array([0.1,0.12,0.18])
    cols = np.array(colors)/255.0
    return np.clip(cols.mean(axis=0), 0, 1)

def make_bg_from_emotions(width, height, selected_emotions, palette, darkness=0.35, lightness=0.85):
    if not selected_emotions:
        base = np.array([0.08,0.09,0.12])
    else:
        base = mix_colors([palette.get(e, (180,180,185)) for e in selected_emotions])
    top = np.clip(base*darkness, 0, 1)
    bottom = np.clip(1 - (1-base)* (1-lightness), 0, 1)
    grad = np.linspace(0,1,height).reshape(height,1,1)
    img = top.reshape(1,1,3)*(1-grad) + bottom.reshape(1,1,3)*grad
    img = (img*255).astype(np.uint8)
    img = np.tile(img,(1,width,1))
    img = np.ascontiguousarray(img)
    return np.array(Image.fromarray(img), dtype=np.float32)/255.0

def bezier_points(p0, p1, p2, p3, t):
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

def draw_gaussian_disc(canvas, cx, cy, radius, color, alpha):
    h, w, _ = canvas.shape
    r = int(max(1, radius))
    x0, x1 = max(0, int(cx - r)), min(w-1, int(cx + r))
    y0, y1 = max(0, int(cy - r)), min(h-1, int(cy + r))
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    sigma2 = (radius*0.6)**2 + 1e-6
    a = np.exp(-dist2/(2*sigma2))*alpha
    col = np.array(color).reshape(1,1,3)
    sub = canvas[y0:y1+1, x0:x1+1, :]
    sub[:] = sub*(1 - a[...,None]) + col*a[...,None]

def draw_ribbon(canvas, rng, color_rgb, width_px, smoothness, flow_variance, alpha, bands_noise=0.15):
    h, w, _ = canvas.shape
    # random endpoints along left/right or top/bottom, then bezier in between
    if rng.random() < 0.5:
        x0, y0 = -0.1*w, rng.uniform(0.15*h, 0.85*h)
        x3, y3 = 1.1*w, rng.uniform(0.15*h, 0.85*h)
    else:
        x0, y0 = rng.uniform(0.1*w, 0.9*w), -0.1*h
        x3, y3 = rng.uniform(0.1*w, 0.9*w), 1.1*h

    # control points
    dx = (x3 - x0)
    dy = (y3 - y0)
    curve_amp = smoothness * 0.35
    x1 = x0 + dx*0.33 + rng.normal(0, abs(dx)*0.2)*curve_amp
    y1 = y0 + dy*0.33 + rng.normal(0, abs(dy)*0.2)*curve_amp
    x2 = x0 + dx*0.66 + rng.normal(0, abs(dx)*0.2)*curve_amp
    y2 = y0 + dy*0.66 + rng.normal(0, abs(dy)*0.2)*curve_amp

    p0 = np.array([x0,y0]); p1 = np.array([x1,y1]); p2 = np.array([x2,y2]); p3 = np.array([x3,y3])

    steps = int(1200 * (0.7 + 0.6*smoothness))
    t = np.linspace(0, 1, steps)
    pts = bezier_points(p0, p1, p2, p3, t)

    # width modulation and micro meander
    base_w = width_px
    for i in range(steps):
        px, py = pts[i]
        jitter = rng.normal(0, 1.0) * flow_variance
        px += jitter
        py += jitter*0.5
        rad = base_w * (0.75 + 0.25*np.sin(i*0.05 + rng.random()*2*np.pi))
        local_alpha = alpha * (0.7 + 0.3*np.cos(i*0.03))
        # gentle noise color richness
        noise_col = (np.array(color_rgb)/255.0) * (0.9 + 0.2*np.sin(i*0.02 + rng.random()))
        noise_col = np.clip(noise_col, 0, 1)
        draw_gaussian_disc(canvas, px, py, rad, noise_col, local_alpha)

def render_ribbon_flow(df, palette, width, height, seed,
                       bands_per_emotion=3, ribbon_width=18.0, smoothness=0.85,
                       flow_variance=1.8, blur_px=2.0, alpha=0.18):
    rng = np.random.default_rng(seed)

    # choose active emotions (or gentle defaults)
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["hope","calm","awe"]

    # background based on selected emotions (no fixed base tone)
    bg = make_bg_from_emotions(width, height, emotions, palette, darkness=0.28, lightness=0.92)
    canvas = bg.copy()

    # render ribbons
    for emo in emotions:
        base_rgb = palette.get(emo, palette.get("mixed",(210,190,140)))
        # jitter for richness and separation
        emo_rgb = jitter_emotion_color(base_rgb, emo, amount=0.10)
        for _ in range(max(1,int(bands_per_emotion))):
            draw_ribbon(canvas, rng, emo_rgb, width_px=ribbon_width,
                        smoothness=smoothness, flow_variance=flow_variance,
                        alpha=alpha)

    # mild global blur to increase silkiness
    out = Image.fromarray((np.clip(canvas,0,1)*255).astype(np.uint8))
    if blur_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur_px))
    return out

# =========================
# UI - Sidebar
# =========================

# ---- 1) Data Source (NewsAPI only)
st.sidebar.header("1) Data Source (NewsAPI only)")
st.sidebar.markdown("**Keyword** (e.g., aurora borealis, space weather, AI, technology)")
keyword = st.sidebar.text_input("", value="")
fetch_btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "aurora")

if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking flow of emotions spreads across the night sky.",
        "Calm conditions create a gentle environment.",
        "Anxiety spreads among investors during unstable market conditions.",
        "A moment of awe as the stream of light dances.",
        "Hope arises as scientific discoveries advance our understanding."
    ]})
    df["timestamp"]=str(date.today())

df["text"]=df["text"].fillna("")

# Sentiment and emotion mapping
sent_df=df["text"].apply(analyze_sentiment).apply(pd.Series)
df=pd.concat([df.reset_index(drop=True),sent_df.reset_index(drop=True)],axis=1)
df["emotion"]=df.apply(classify_emotion_expanded,axis=1)

# ---- 2) Emotion Mapping
st.sidebar.header("2) Emotion Mapping")
cmp_min, cmp_max = st.sidebar.slider("Compound Range", -1.0,1.0,(-1.0,1.0),0.01)

init_palette_state()
base_palette = get_active_palette()

available_emotions = sorted(df["emotion"].unique().tolist())
custom_emotions = sorted(set(base_palette.keys()) - set(DEFAULT_RGB.keys()))
all_emotions_for_ui = list(ALL_EMOTIONS) + [e for e in custom_emotions if e not in ALL_EMOTIONS]

def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = base_palette.get(e, (0, 0, 0))
    return f"{e} (Custom {r},{g},{b})"

options_labels = [_label_emotion(e) for e in all_emotions_for_ui]
default_labels = [_label_emotion(e) for e in available_emotions] if available_emotions else options_labels
selected_labels = st.sidebar.multiselect("Show Emotions:", options_labels, default=default_labels)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

df = df[(df["emotion"].isin(selected_emotions)) & (df["compound"]>=cmp_min) & (df["compound"]<=cmp_max)]

# ---- 3) Ribbon Flow Engine (only)
st.sidebar.header("3) Ribbon Flow Engine")
bands = st.sidebar.slider("Bands per Emotion", 1, 6, 3, 1)
ribbon_width = st.sidebar.slider("Ribbon Width (px)", 6.0, 40.0, 22.0, 0.5)
smoothness = st.sidebar.slider("Curve Smoothness", 0.3, 1.2, 0.90, 0.01)
flow_variance = st.sidebar.slider("Flow Randomness", 0.2, 4.0, 2.0, 0.1)
blur_px = st.sidebar.slider("Silkiness Blur (px)", 0.0, 10.0, 2.5, 0.1)
alpha_ribbon = st.sidebar.slider("Ribbon Opacity", 0.05, 0.45, 0.22, 0.01)
img_brightness = st.sidebar.slider("Base Brightness", 0.8, 1.6, 1.10, 0.02)

# ---- 4) Cinematic Color System
st.sidebar.header("4) Cinematic Color System")
palette_mode = st.sidebar.selectbox(
    "Palette Preset",
    list(CINEMATIC_PRESETS.keys()),
    index=list(CINEMATIC_PRESETS.keys()).index("Planetary (Soft)")
)
exp = st.sidebar.slider("Exposure (stops)", -0.3, 1.7, 0.50, 0.01)
contrast = st.sidebar.slider("Contrast", 0.70, 1.90, 1.20, 0.01)
saturation = st.sidebar.slider("Saturation", 0.70, 2.00, 1.25, 0.01)
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40, 0.95, 0.01)
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50, 0.40, 0.01)

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature (Blue ↔ Red)", -1.0, 1.0, 0.05, 0.01)
tint = st.sidebar.slider("Tint (Green ↔ Magenta)", -1.0, 1.0, 0.02, 0.01)

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, 0.10, 0.01)
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, 0.08, 0.01)
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, 0.16, 0.01)
hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, 0.12, 0.01)
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, 0.10, 0.01)
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, 0.08, 0.01)
tone_balance = st.sidebar.slider("Tone Balance (Shadows ↔ Highlights)", -1.0, 1.0, 0.0, 0.01)

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius (px)", 0.0, 20.0, 10.0, 0.5)
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0, 0.55, 0.01)
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8, 0.18, 0.01)

# ---- 5) Auto Brightness Compensation
st.sidebar.header("5) Auto Brightness Compensation")
auto_bright = st.sidebar.checkbox("Enable Auto Brightness", value=True)
target_mean = st.sidebar.slider("Target Mean Luminance", 0.30, 0.70, 0.50, 0.01)
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, 0.85, 0.05)
abc_black = st.sidebar.slider("Black Point Percentile", 0.00, 0.20, 0.06, 0.01)
abc_white = st.sidebar.slider("White Point Percentile", 0.80, 1.00, 0.995, 0.001)
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0, 2.4, 0.05)

# ---- 6) Custom Palette (RGB)
st.sidebar.header("6) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette",value=st.session_state["use_csv_palette"])
st.session_state["use_csv_palette"]=use_csv

with st.sidebar.expander("Add Custom Emotion",False):
    col1,col2,col3,col4=st.columns([1.8,1,1,1])
    name=col1.text_input("Emotion")
    r=col2.number_input("R",0,255,210)
    g=col3.number_input("G",0,255,190)
    b=col4.number_input("B",0,255,140)
    if st.button("Add"):
        add_custom_emotion(name,r,g,b)
    show = st.session_state.get("custom_palette",{})
    if show:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in show.items()]),
                     use_container_width=True,height=150)

with st.sidebar.expander("Import / Export Palette CSV",False):
    up = st.file_uploader("Import CSV",type=["csv"])
    if up is not None:
        import_palette_csv(up)
    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state["custom_palette"])
    if st.session_state["use_csv_palette"]:
        pal = dict(st.session_state["custom_palette"])
    if pal:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]),
                     use_container_width=True,height=160)
        dl = export_palette_csv(pal)
        st.download_button("Download CSV",data=dl,file_name="palette.csv",mime="text/csv")

# ---- 7) Output
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All"):
    st.session_state.clear()
    st.rerun()

# =========================
# DRAW SECTION
# =========================
left, right = st.columns([0.60,0.40])

with left:
    st.subheader("Flowing Ribbon Visual")

    if df.empty:
        st.warning("No data points under current filters.")
    else:
        # palette preset to enrich base hues
        working_palette, preset_temp, preset_tint = apply_palette_preset(base_palette, palette_mode)

        # render ribbon flow (base domain: sRGB 0..1)
        img = render_ribbon_flow(
            df=df,
            palette=working_palette,
            width=1500,
            height=850,
            seed=np.random.randint(0, 999999),
            bands_per_emotion=bands,
            ribbon_width=ribbon_width,
            smoothness=smoothness,
            flow_variance=flow_variance,
            blur_px=blur_px,
            alpha=alpha_ribbon
        )

        # base brightness lift (pre-grade)
        arr = np.array(img).astype(np.float32)/255.0
        arr = np.clip(arr * img_brightness, 0, 1)

        # ======== Cinematic Color Pipeline ========
        lin = srgb_to_linear(arr)
        lin = lin * (2.0 ** exp)
        lin = apply_white_balance(lin, temp + preset_temp, tint + preset_tint)
        lin = highlight_rolloff(lin, roll)
        arr = linear_to_srgb(np.clip(lin, 0, 6))
        arr = np.clip(filmic_tonemap(arr*1.25), 0, 1)
        arr = adjust_contrast(arr, contrast)
        arr = adjust_saturation(arr, saturation)
        arr = gamma_correct(arr, gamma_val)
        arr = split_tone(arr, (sh_r, sh_g, sh_b), (hi_r, hi_g, hi_b), tone_balance)

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
        arr = ensure_colorfulness(arr, min_sat=0.20, boost=1.22)

        final_img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="RGB")
        buf=BytesIO(); final_img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf,use_column_width=True)
        st.download_button("Download PNG",data=buf,file_name="ribbon_flow_cinematic.png",mime="image/png")

with right:
    st.subheader("Data & Emotion")
    df2=df.copy()
    df2["emotion_display"]=df2["emotion"].apply(lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})")
    cols=["text","emotion_display","compound","pos","neu","neg"]
    if "timestamp" in df.columns: cols.insert(1,"timestamp")
    if "source" in df.columns: cols.insert(2,"source")
    st.dataframe(df2[cols],use_container_width=True,height=600)

st.markdown("---")
st.caption("© 2025 Emotional Ribbon Flow — Cinematic Edition")
