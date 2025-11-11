# app.py
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
st.set_page_config(page_title="Emotional Ribbon Flow — Wang Xinru — Final Project",
                   page_icon="🎞️", layout="wide")
st.title("🎞️ Emotional Ribbon Flow — Wang Xinru — Final Project")

# ============== Instructions ============
with st.expander("Instructions", expanded=False):
    st.markdown(
        """
### How to Use

Turn live news emotions into **blue-white print-style ribbon art** with a **cinematic color system** and **Auto Brightness Compensation**.

1) **Fetch Data** — NewsAPI only. Provide a keyword (e.g., *aurora borealis*, *AI*, *technology*).  
2) **Emotion Mapping** — VADER → curated emotions; filter by compound range and choose which emotions to visualize.  
3) **Ribbon Flow Renderer** — control ribbon count/width/smoothness/opacity and flow variance; theme follows emotion.  
4) **Cinematic Color System** — exposure/contrast/gamma/saturation, white balance, split-toning, bloom, vignette, Auto-Brightness.  
5) **Palette** — one fixed color per emotion; add custom RGB; CSV import/export; background tone adapts to dominant emotions.  
6) **Download** — save PNG.
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
# Emotion color (planet-like, can be overridden)
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

# Background themes (used as fallbacks; final bg will adapt to emotions)
THEMES = {
    "Deep Night": ((0.02, 0.03, 0.08), (0.0, 0.0, 0.0)),
    "Polar Twilight": ((0.08, 0.10, 0.18), (0.02, 0.02, 0.04)),
    "Dawn Haze": ((0.12, 0.10, 0.12), (0.01, 0.01, 0.02)),
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
# Utilities (color & bg)
# =========================
def lerp(a, b, t):
    return a*(1-t) + b*t

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def rgb255_to_f(rgb):
    return np.array(rgb, dtype=np.float32)/255.0

def f_to_rgb255(f):
    return tuple((clamp01(np.array(f))*255).astype(np.uint8).tolist())

def color_lighten(rgb, amt=0.25):
    f = rgb255_to_f(rgb)
    return f_to_rgb255(lerp(f, np.ones(3, dtype=np.float32), amt))

def color_darken(rgb, amt=0.25):
    f = rgb255_to_f(rgb)
    return f_to_rgb255(lerp(f, np.zeros(3, dtype=np.float32), amt))

def emotions_to_bg_top_bottom(palette, emo_counts, fallback_theme="Polar Twilight"):
    """Choose background top/bottom from weighted emotions."""
    if len(emo_counts) == 0:
        top_f, bot_f = THEMES[fallback_theme]
        return f_to_rgb255(top_f), f_to_rgb255(bot_f)
    keys = list(emo_counts.keys())
    weights = np.array(list(emo_counts.values()), dtype=np.float32)
    weights = weights/weights.sum()
    cols = np.array([rgb255_to_f(palette.get(k, (180,180,185))) for k in keys])
    mean = (cols * weights[:,None]).sum(axis=0)
    top = clamp01(lerp(mean, np.ones(3), 0.15))
    bottom = clamp01(lerp(mean, np.zeros(3), 0.35))
    return f_to_rgb255(top), f_to_rgb255(bottom)

def vertical_gradient(width, height, top_rgb, bottom_rgb, brightness=1.0):
    t = rgb255_to_f(top_rgb)*brightness
    b = rgb255_to_f(bottom_rgb)*brightness
    grad = np.linspace(0,1,height).reshape(height,1,1)
    img = t.reshape(1,1,3)*(1-grad) + b.reshape(1,1,3)*grad
    img = (img*255).astype(np.uint8)
    img = np.tile(img,(1,width,1))
    img = np.ascontiguousarray(img)
    return Image.fromarray(img, mode="RGB")

# =========================
# Bezier & Ribbon drawing
# =========================
def cubic_bezier(p0, p1, p2, p3, t):
    """Return point(s) on cubic Bezier for scalar or 1D-array t."""
    t = np.asarray(t, dtype=np.float32)
    u = 1.0 - t
    return (u*u*u)[:,None]*p0 + (3*u*u*t)[:,None]*p1 + (3*u*t*t)[:,None]*p2 + (t*t*t)[:,None]*p3

def bezier_polyline(p0, p1, p2, p3, steps=600):
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    pts = cubic_bezier(p0, p1, p2, p3, t)  # (steps, 2)
    return [tuple(map(float, p)) for p in pts]

def draw_ribbon(img, rng, base_rgb, width_px=18, smoothness=0.75,
                flow_variance=0.35, alpha=0.85, taper=True):
    """
    Draw one flowing ribbon: smooth cubic bezier with stable direction,
    many parallel sub-lines to simulate a wide, soft ribbon.
    """
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Anchor points
    x0 = rng.uniform(0.05*W, 0.95*W)
    y0 = rng.uniform(0.05*H, 0.95*H)

    # Direction
    theta = rng.uniform(-np.pi, np.pi)
    length = rng.uniform(0.7*min(W,H), 1.2*min(W,H))

    dx, dy = np.cos(theta), np.sin(theta)

    p0 = np.array([x0, y0], dtype=np.float32)
    p3 = np.array([x0 + dx*length, y0 + dy*length], dtype=np.float32)

    # Controls with smoothness parameter
    def jitter(scale):
        return np.array([rng.normal(0, scale), rng.normal(0, scale)], dtype=np.float32)

    ctrl_scale = (0.35 + 0.45*(1.0-smoothness)) * length
    p1 = p0 + np.array([-dy, dx], dtype=np.float32) * rng.uniform(-0.35, 0.35) * ctrl_scale + jitter(flow_variance*60)
    p2 = p3 + np.array([ dy,-dx], dtype=np.float32) * rng.uniform(-0.35, 0.35) * ctrl_scale + jitter(flow_variance*60)

    poly = bezier_polyline(p0, p1, p2, p3, steps=900)

    # Multiple strokes across the width with soft alpha falloff
    layers = max(6, int(width_px//3))
    center_alpha = int(255*alpha)
    for i in range(-layers, layers+1):
        t = i/layers  # -1..1
        w = int(max(1, width_px * (1.0 - 0.65*abs(t))))
        a = int(center_alpha * (1.0 - 0.85*abs(t)))
        col = (*base_rgb, a)
        draw.line(poly, fill=col, width=w, joint="curve")

    # Optional taper blur to soften
    blur_rad = max(1, int(0.08*width_px))
    if blur_rad > 0:
        img_blur = img.filter(ImageFilter.GaussianBlur(radius=blur_rad))
        img.alpha_composite(img_blur, dest=(0,0))

def add_paper_grain(rgb_img, amount=0.08):
    """Add small white-dust grain like in your examples."""
    W, H = rgb_img.size
    arr = np.array(rgb_img).astype(np.float32)/255.0
    rng = np.random.default_rng(2025)
    noise = rng.random((H, W, 1)).astype(np.float32)
    dots = (noise > 0.995).astype(np.float32) * amount * 4.0
    arr = clamp01(arr + dots)
    return Image.fromarray((arr*255).astype(np.uint8), mode="RGB")

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

def apply_white_balance(img, temp, tint):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    r *= (1.0 + 0.10*temp)
    b *= (1.0 - 0.10*temp)
    g *= (1.0 + 0.08*tint)
    r *= (1.0 - 0.06*tint)
    b *= (1.0 - 0.02*tint)
    out = np.stack([r,g,b], axis=-1)
    return clamp01(out)

def adjust_contrast(img, c):
    return clamp01((img - 0.5)*c + 0.5)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = lum[...,None]
    return clamp01(lum + (img - lum)*s)

def gamma_correct(img, g):
    return clamp01(img ** (1.0/max(g,1e-6)))

def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5)
    threshold = 0.8
    mask = np.clip((img - threshold)/(1e-6 + 1.0 - threshold), 0, 1)
    out = img*(1 - mask) + (threshold + (img-threshold)/(1.0 + 4.0*t*mask))*mask
    return clamp01(out)

def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = (lum - lum.min())/(lum.max()-lum.min()+1e-6)
    sh = np.clip(1.0 - lum + 0.5*(1-balance), 0, 1)[...,None]
    hi = np.clip(lum + 0.5*(1+balance) - 0.5, 0, 1)[...,None]
    sh_col = np.array(sh_rgb).reshape(1,1,3)
    hi_col = np.array(hi_rgb).reshape(1,1,3)
    out = clamp01(img + sh*sh_col*0.25 + hi*hi_col*0.25)
    return out

def apply_bloom(img, radius=6.0, intensity=0.6):
    pil = Image.fromarray((clamp01(img)*255).astype(np.uint8), mode="RGB")
    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32)/255.0
        out = clamp01(img*(1-intensity) + b*intensity)
        return out
    return img

def apply_vignette(img, strength=0.25):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2)/(w/2); yy = (yy - h/2)/(h/2)
    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength*(r**1.5), 0.0, 1.0)
    return clamp01(img * mask[...,None])

def ensure_colorfulness(img, min_sat=0.15, boost=1.25):
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)
    if float(sat.mean()) < min_sat:
        return adjust_saturation(img, boost)
    return img

# Presets (affect palette modulation, WB seed)
CINEMATIC_PRESETS = {
    "Planetary (Soft)": {"mult": (1.00, 1.00, 1.00), "sat": 1.00, "temp": 0.00, "tint": 0.00},
    "Cinematic Cool":  {"mult": (0.95, 1.02, 1.08), "sat": 1.05, "temp": -0.20, "tint": 0.02},
    "Cinematic Warm":  {"mult": (1.08, 1.02, 0.95), "sat": 1.05, "temp": 0.20,  "tint": -0.02},
    "Neon Arctic":     {"mult": (0.90, 1.05, 1.15), "sat": 1.20, "temp": -0.30, "tint": 0.05},
    "Sunset Storm":    {"mult": (1.15, 1.03, 0.92), "sat": 1.18, "temp": 0.25,  "tint": 0.06},
    "Pastel Dream":    {"mult": (1.03, 1.03, 1.03), "sat": 0.92, "temp": 0.05,  "tint": 0.05},
    "Deep Space":      {"mult": (0.95, 0.98, 1.05), "sat": 0.95, "temp": -0.10, "tint": 0.00},
}

def apply_palette_preset(base_palette: dict, preset_name: str):
    p = CINEMATIC_PRESETS.get(preset_name, CINEMATIC_PRESETS["Planetary (Soft)"])
    mult = np.array(p["mult"])
    sat = p["sat"]
    out = {}
    for k, rgb in base_palette.items():
        col = np.array(rgb)/255.0
        col = clamp01(col * mult)
        # saturation tweak in tiny 1x1 "image"
        c = adjust_saturation(col.reshape(1,1,3), sat)[0,0,:]
        out[k] = tuple((c*255).astype(int).tolist())
    return out, p["temp"], p["tint"]

# =========================
# Auto Brightness Compensation
# =========================
def auto_brightness_compensation(img_arr, target_mean=0.45, strength=0.9,
                                 black_point_pct=0.06, white_point_pct=0.995,
                                 max_gain=2.4):
    arr = clamp01(img_arr.astype(np.float32))
    lin = srgb_to_linear(arr)
    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-6: wp = bp + 1e-3
    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    remap_gain = np.clip(strength, 0, 1)
    Y_final = (1-remap_gain)*Y + remap_gain*Y_remap
    meanY = max(float(Y_final.mean()), 1e-4)
    gain = np.clip(target_mean / meanY, 1.0/max_gain, max_gain)
    lin *= gain
    Y2 = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    blend = 0.65*remap_gain
    Y_mix = (1-blend)*Y2 + blend*np.clip(Y_final*gain, 0, 2.5)
    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = clamp01(lin * ratio[...,None])
    out = filmic_tonemap(clamp01(lin))
    out = clamp01(out)
    out = linear_to_srgb(out)
    return clamp01(out)

# =========================
# Ribbon Flow Renderer (main)
# =========================
def render_ribbon_flow(df, palette, theme_name,
                       width=1500, height=2000,   # tall poster works well for this style
                       seed=1234,
                       ribbons_per_emotion=3,
                       ribbon_width=22,
                       smoothness=0.85,
                       flow_variance=0.30,
                       alpha=0.90,
                       bg_brightness=1.0,
                       add_grain=True):
    rng = np.random.default_rng(seed)

    # Background color from emotions
    counts = df["emotion"].value_counts().to_dict()
    top_rgb, bot_rgb = emotions_to_bg_top_bottom(palette, counts, fallback_theme=theme_name)

    bg = vertical_gradient(width, height, top_rgb, bot_rgb, brightness=bg_brightness).convert("RGBA")
    canvas = Image.new("RGBA", (width, height))
    canvas.alpha_composite(bg)

    # Sort emotions by frequency (dominant first)
    emos_sorted = list(df["emotion"].value_counts().index)
    if not emos_sorted:
        emos_sorted = ["calm", "trust", "awe"]

    # Draw ribbons per emotion
    for emo in emos_sorted:
        base_col = palette.get(emo, (200,200,200))
        # To mimic your references: main lines are near white, with subtle tint from emotion
        tint = rgb255_to_f(base_col)
        near_white = clamp01(lerp(np.ones(3), tint, 0.25))  # mostly white with emotion tint
        emo_rgb = f_to_rgb255(near_white)

        n = max(1, int(ribbons_per_emotion))
        for _ in range(n):
            w = int(ribbon_width * rng.uniform(0.85, 1.15))
            a = float(alpha) * rng.uniform(0.85, 1.05)
            sm = float(np.clip(smoothness * rng.uniform(0.9, 1.1), 0.4, 0.98))
            fv = float(np.clip(flow_variance * rng.uniform(0.85, 1.2), 0.05, 0.6))
            draw_ribbon(canvas, rng, emo_rgb, width_px=w,
                        smoothness=sm, flow_variance=fv, alpha=a, taper=True)

    out = canvas.convert("RGB")
    if add_grain:
        out = add_paper_grain(out, amount=0.06)
    return out

# =========================
# UI (kept from your aurora app, adjusted for Ribbon Flow)
# =========================

# ---- 1) Data Source (NewsAPI only)
st.sidebar.header("1) Data Source (NewsAPI only)")
st.sidebar.markdown("**Keyword** *(e.g., aurora borealis, space weather, AI, technology)*")
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
        "A breathtaking aurora illuminated the northern sky last night.",
        "Calm atmospheric conditions create a beautiful environment.",
        "Anxiety spreads among investors during unstable market conditions.",
        "A moment of awe as the sky shines with green auroral light.",
        "Hope arises as scientific discoveries advance our understanding."
    ]})
    df["timestamp"]=str(date.today())

df["text"]=df["text"].fillna("")

# Sentiment
sent_df=df["text"].apply(analyze_sentiment).apply(pd.Series)
df=pd.concat([df.reset_index(drop=True),sent_df.reset_index(drop=True)],axis=1)
df["emotion"]=df.apply(classify_emotion_expanded,axis=1)

# ---- 2) Emotion Filter
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
df=df[(df["emotion"].isin(selected_emotions))&(df["compound"]>=cmp_min)&(df["compound"]<=cmp_max)]

# ---- 3) Ribbon Flow Renderer
st.sidebar.header("3) Ribbon Flow Renderer")
theme_name = st.sidebar.selectbox("Background Theme (fallback)", list(THEMES.keys()), index=1)
bg_brightness = st.sidebar.slider("Background Brightness", 0.6, 1.8, 1.1, 0.05)

ribbons_per_emotion = st.sidebar.slider("Ribbons per Emotion", 1, 8, 4, 1)
ribbon_width = st.sidebar.slider("Ribbon Width (px)", 6, 48, 22, 1)
smoothness = st.sidebar.slider("Smoothness (0=wild, 1=very smooth)", 0.4, 0.98, 0.85, 0.01)
flow_variance = st.sidebar.slider("Flow Variance", 0.05, 0.60, 0.30, 0.01)
alpha_ribbon = st.sidebar.slider("Ribbon Opacity", 0.40, 1.00, 0.90, 0.01)
add_grain = st.sidebar.checkbox("Paper Grain", value=True)

# ---- 4) Cinematic Color System
st.sidebar.header("4) Cinematic Color System")
palette_mode = st.sidebar.selectbox(
    "Palette Preset",
    list(CINEMATIC_PRESETS.keys()),
    index=list(CINEMATIC_PRESETS.keys()).index("Planetary (Soft)")
)
exp = st.sidebar.slider("Exposure (stops)", -0.5, 1.5, 0.35, 0.01)
contrast = st.sidebar.slider("Contrast", 0.70, 1.80, 1.18, 0.01)
saturation = st.sidebar.slider("Saturation", 0.70, 1.90, 1.18, 0.01)
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40, 0.95, 0.01)
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50, 0.45, 0.01)

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature (Blue ↔ Red)", -1.0, 1.0, 0.05, 0.01)
tint = st.sidebar.slider("Tint (Green ↔ Magenta)", -1.0, 1.0, 0.02, 0.01)

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, 0.10, 0.01)
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, 0.06, 0.01)
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, 0.14, 0.01)
hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, 0.12, 0.01)
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, 0.10, 0.01)
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, 0.08, 0.01)
tone_balance = st.sidebar.slider("Tone Balance (Shadows ↔ Highlights)", -1.0, 1.0, 0.0, 0.01)

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius (px)", 0.0, 18.0, 9.0, 0.5)
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0, 0.48, 0.01)
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8, 0.20, 0.01)

# ---- 5) Auto Brightness Compensation
st.sidebar.header("5) Auto Brightness Compensation")
auto_bright = st.sidebar.checkbox("Enable Auto Brightness", value=True)
target_mean = st.sidebar.slider("Target Mean Luminance", 0.25, 0.65, 0.48, 0.01)
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, 0.85, 0.05)
abc_black = st.sidebar.slider("Black Point Percentile", 0.00, 0.20, 0.06, 0.01)
abc_white = st.sidebar.slider("White Point Percentile", 0.80, 1.00, 0.995, 0.001)
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0, 2.2, 0.05)

# ---- 6) Custom Palette (RGB)
st.sidebar.header("6) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette", value=st.session_state["use_csv_palette"])
st.session_state["use_csv_palette"] = use_csv

with st.sidebar.expander("Add Custom Emotion", False):
    col1, col2, col3, col4 = st.columns([1.8,1,1,1])
    name = col1.text_input("Emotion")
    r = col2.number_input("R", 0, 255, 210)
    g = col3.number_input("G", 0, 255, 190)
    b = col4.number_input("B", 0, 255, 140)
    if st.button("Add"):
        add_custom_emotion(name, r, g, b)
    show = st.session_state.get("custom_palette", {})
    if show:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in show.items()]),
                     use_container_width=True, height=150)

with st.sidebar.expander("Import / Export Palette CSV", False):
    up = st.file_uploader("Import CSV", type=["csv"])
    if up is not None:
        import_palette_csv(up)
    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state["custom_palette"])
    if st.session_state["use_csv_palette"]:
        pal = dict(st.session_state["custom_palette"])
    if pal:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]),
                     use_container_width=True, height=160)
        dl = export_palette_csv(pal)
        st.download_button("Download CSV", data=dl, file_name="palette.csv", mime="text/csv")

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
    st.subheader("🎞️ Ribbon Flow")

    if df.empty:
        st.warning("No data points under current filters.")
    else:
        # palette preset first
        working_palette, preset_temp, preset_tint = apply_palette_preset(get_active_palette(), palette_mode)

        # render ribbons
        img = render_ribbon_flow(
            df=df, palette=working_palette, theme_name=theme_name,
            width=1500, height=900,
            seed=np.random.randint(0, 999999),
            ribbons_per_emotion=ribbons_per_emotion,
            ribbon_width=ribbon_width,
            smoothness=smoothness,
            flow_variance=flow_variance,
            alpha=alpha_ribbon,
            bg_brightness=bg_brightness,
            add_grain=add_grain
        )

        # ===== Cinematic Color Pipeline =====
        arr = np.array(img).astype(np.float32)/255.0

        # exposure in linear
        lin = srgb_to_linear(arr)
        lin = lin * (2.0 ** exp)

        # white balance: preset + user
        lin = apply_white_balance(lin, temp + preset_temp, tint + preset_tint)

        # highlight roll-off
        lin = highlight_rolloff(lin, roll)

        # back to display + filmic clamp
        arr = linear_to_srgb(clamp01(lin))
        arr = clamp01(filmic_tonemap(arr*1.15))

        # creative grading
        arr = adjust_contrast(arr, contrast)
        arr = adjust_saturation(arr, saturation)
        arr = gamma_correct(arr, gamma_val)
        arr = split_tone(arr, (sh_r, sh_g, sh_b), (hi_r, hi_g, hi_b), tone_balance)

        # Auto Brightness after creative grade
        if auto_bright:
            arr = auto_brightness_compensation(
                arr, target_mean=target_mean, strength=abc_strength,
                black_point_pct=abc_black, white_point_pct=abc_white, max_gain=abc_max_gain
            )

        # bloom & vignette
        arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
        arr = apply_vignette(arr, strength=vignette_strength)

        # ensure colorfulness
        arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.20)

        final_img = Image.fromarray((clamp01(arr)*255).astype(np.uint8), mode="RGB")

        buf = BytesIO(); final_img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf, use_column_width=True)
        st.download_button("💾 Download PNG", data=buf, file_name="ribbon_flow.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotion")
    df2 = df.copy()
    df2["emotion_display"]=df2["emotion"].apply(lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})")
    cols = ["text","emotion_display","compound","pos","neu","neg"]
    if "timestamp" in df.columns: cols.insert(1,"timestamp")
    if "source" in df.columns: cols.insert(2,"source")
    st.dataframe(df2[cols], use_container_width=True, height=600)

st.markdown("---")
st.caption("© 2025 Emotional Ribbon Flow — Cinematic Edition")
