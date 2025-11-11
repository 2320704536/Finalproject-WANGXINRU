# app.py
# 🎐 Emotional Ribbon — Final High-Brightness Edition
# - Ribbon Flow renderer only (no aurora bands)
# - NewsAPI → VADER → curated emotions
# - Auto-select top 3 emotions (you can still add/remove manually)
# - Background color derives from the most frequent emotion
# - Cinematic Color System + Auto Brightness Compensation
# - Per-emotion diverse colors (no single blue/white bias)
# - Palette CSV import/export + custom RGB
# - PNG download

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw
import colorsys
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Ribbon — Wang Xinru — Final Project", page_icon="🎐", layout="wide")
st.title("🎐 Emotional Ribbon — Wang Xinru — Final Project")

# ✅ Instructions section (concise)
with st.expander("How to Use", expanded=False):
    st.markdown("""
This app turns live news emotions into **Ribbon Flow** graphics with a **cinematic color system** and **Auto Brightness Compensation** so images are bright and colorful.

1) **Fetch Data** — NewsAPI only (enter keyword).  
2) **Emotion Mapping** — VADER → curated emotions; **auto-selects top 3** (you can edit).  
3) **Ribbon Engine** — adjust ribbons/length/width/flow/softness.  
4) **Cinematic Color** — exposure/contrast/gamma/saturation, WB, split-toning, bloom, vignette, **Auto Brightness**.  
5) **Palette** — fixed per emotion; add custom RGB; CSV import/export.  
6) **Download** — save PNG.
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
# Colors
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

# (old gradient themes kept for fallback if needed somewhere else)
THEMES = {
    "Deep Night": ((0.02, 0.03, 0.08), (0.00, 0.00, 0.00)),
    "Polar Twilight": ((0.06, 0.08, 0.16), (0.00, 0.00, 0.00)),
    "Dawn Haze": ((0.10, 0.08, 0.12), (0.00, 0.00, 0.00)),
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
# Helpers: gradients & noise
# =========================
def vertical_gradient(width, height, top_rgb, bottom_rgb, brightness=1.0):
    t = np.array(top_rgb)*brightness
    b = np.array(bottom_rgb)*brightness
    grad = np.linspace(0,1,height).reshape(height,1,1)
    img = t.reshape(1,1,3)*(1-grad) + b.reshape(1,1,3)*grad
    img = (img*255).astype(np.uint8)
    img = np.tile(img,(1,width,1))
    img = np.ascontiguousarray(img)
    return Image.fromarray(img)

def fbm_noise(h, w, rng, octaves=5, base_scale=96, persistence=0.55, lacunarity=2.0):
    acc = np.zeros((h,w), dtype=np.float32)
    amp = 1.0
    scale = base_scale
    for _ in range(octaves):
        gh = max(1, h//max(1,scale)); gw = max(1, w//max(1,scale))
        g = rng.random((gh,gw)).astype(np.float32)
        layer = np.array(
            Image.fromarray((g*255).astype(np.uint8)).resize((w,h), Image.BICUBIC),
            dtype=np.float32
        )/255.
        acc += layer*amp
        amp *= persistence
        scale = max(1, int(scale/lacunarity))
    acc -= acc.min()
    if acc.max()>1e-6: acc /= acc.max()
    return acc

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
    r *= (1.0 + 0.10*temp); b *= (1.0 - 0.10*temp)
    g *= (1.0 + 0.08*tint); r *= (1.0 - 0.06*tint); b *= (1.0 - 0.02*tint)
    out = np.stack([r,g,b], axis=-1)
    return np.clip(out, 0, 1)

def adjust_contrast(img, c):   return np.clip((img - 0.5)*c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = lum[...,None]; return np.clip(lum + (img - lum)*s, 0, 1)

def gamma_correct(img, g):     return np.clip(img ** (1.0/g), 0, 1)

def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5); threshold = 0.8
    mask = np.clip((img - threshold)/(1e-6 + 1.0 - threshold), 0, 1)
    out = img*(1 - mask) + (threshold + (img-threshold)/(1.0 + 4.0*t*mask))*mask
    return np.clip(out, 0, 1)

def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    lum = (lum - lum.min())/(lum.max()-lum.min()+1e-6)
    sh = np.clip(1.0 - lum + 0.5*(1-balance), 0, 1)[...,None]
    hi = np.clip(lum + 0.5*(1+balance) - 0.5, 0, 1)[...,None]
    sh_col = np.array(sh_rgb).reshape(1,1,3); hi_col = np.array(hi_rgb).reshape(1,1,3)
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

def ensure_colorfulness(img, min_sat=0.15, boost=1.25):
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r,g), b); mn = np.minimum(np.minimum(r,g), b)
    sat = (mx - mn) / (mx + 1e-6)
    if sat.mean() < min_sat: return adjust_saturation(img, boost)
    return img

# =========================
# Cinematic palettes
# =========================
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
    mult = np.array(p["mult"]); sat = p["sat"]
    out = {}
    for k, rgb in base_palette.items():
        col = np.array(rgb)/255.0
        col = np.clip(col * mult, 0, 1)
        col = adjust_saturation(col.reshape(1,1,3), sat)[0,0,:]
        out[k] = tuple((col*255).astype(int).tolist())
    return out, p["temp"], p["tint"]

def jitter_emotion_color(rgb, seed_key, amount=0.035):
    rng = np.random.default_rng(abs(hash((seed_key, "ribbon")))% (2**32))
    jitter = (rng.random(3)-0.5)*2*amount
    col = np.clip(np.array(rgb)/255.0 + jitter, 0, 1)
    return tuple((col*255).astype(int).tolist())

# =========================
# Ribbon Flow — colors & field
# =========================
def rgb_to_hsv01(rgb):
    r, g, b = [c/255.0 for c in rgb]
    return colorsys.rgb_to_hsv(r, g, b)

def hsv01_to_rgb_tuple(h, s, v):
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def make_color_variants(rgb):
    """Create 5 diverse variants: base, lighter, darker, complementary, analogous."""
    h, s, v = rgb_to_hsv01(rgb)
    variants = []
    # base
    variants.append(tuple(np.clip(np.array(rgb), 0, 255).astype(int)))
    # lighter (more V, a bit less S)
    variants.append(hsv01_to_rgb_tuple(h, np.clip(s*0.85, 0, 1), np.clip(v*1.12, 0, 1)))
    # darker (less V, maybe slightly more S)
    variants.append(hsv01_to_rgb_tuple(h, np.clip(s*1.05, 0, 1), np.clip(v*0.78, 0, 1)))
    # complementary (h+0.5)
    variants.append(hsv01_to_rgb_tuple((h+0.5)%1.0, s, v))
    # analogous (±0.06)
    variants.append(hsv01_to_rgb_tuple((h+0.06)%1.0, s, v))
    return variants

def generate_flow_field(h, w, rng, scale=180, octaves=5):
    noise = fbm_noise(h, w, rng, octaves=octaves, base_scale=scale, persistence=0.6, lacunarity=2.0)
    angle = noise * 2*np.pi  # [0, 2pi)
    return angle

def draw_polyline(canvas: Image.Image, pts, color_rgb, width, alpha=220, blur_px=0):
    w, h = canvas.size
    layer = Image.new("RGBA", (w,h), (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")
    col = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]), int(alpha))
    if len(pts) >= 2:
        d.line(pts, fill=col, width=int(width), joint="curve")
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    canvas.alpha_composite(layer)

import colorsys

def derive_bg_from_emotion(rgb, brightness=1.15):
    """Beautiful pastel gradient background from emotion color (no harsh edges)."""
    base = np.array(rgb)/255.0
    h, s, v = colorsys.rgb_to_hsv(base[0], base[1], base[2])

    # Softer, pastel top
    top_v = min(1.0, v * 1.25 + 0.07)
    top_s = max(0.12, s * 0.65)      # pastel → reduce saturation
    top = colorsys.hsv_to_rgb(h, top_s, top_v)

    # Very smooth bottom (slightly deeper, but safe)
    bot_v = max(0.12, v * 0.82)
    bot_s = max(0.20, s * 0.78)
    bottom = colorsys.hsv_to_rgb(h, bot_s, bot_v)

    top = np.array(top) * brightness
    bottom = np.array(bottom) * (brightness * 0.92)

    return tuple(np.clip(top,0,1).tolist()), tuple(np.clip(bottom,0,1).tolist())


def render_ribbons(df, palette,
                   width=1500, height=850, seed=12345,
                   emotions_in_use=None,
                   ribbons_per_emotion=14, steps=420, step_len=2.2,
                   stroke_width=4, curve_noise=0.30,
                   ribbon_alpha=225, stroke_blur=0,
                   bg_brightness=1.15):
    rng = np.random.default_rng(seed)

    # pick dominant emotion for background
    counts = df["emotion"].value_counts()
    dominant = counts.index[0] if len(counts)>0 else "calm"
    dom_rgb = palette.get(dominant, (180, 200, 220))
    top_rgb, bot_rgb = derive_bg_from_emotion(dom_rgb, brightness=bg_brightness)
    bg = vertical_gradient(width, height, top_rgb, bot_rgb, brightness=1.0).convert("RGBA")

    # canvas
    canvas = Image.new("RGBA", (width, height), (0,0,0,0))
    base = Image.new("RGBA", (width, height))
    base.paste(bg)

    # flow field
    angle = generate_flow_field(height, width, rng, scale=160, octaves=5)

    # emotions to render
    if emotions_in_use is None or len(emotions_in_use)==0:
        emotions_in_use = ["calm","awe","trust"]

    # draw
    for emo in emotions_in_use:
        raw_rgb = palette.get(emo, palette.get("mixed", (210,190,140)))
        raw_rgb = jitter_emotion_color(raw_rgb, emo, amount=0.03)
        variants = make_color_variants(raw_rgb)  # list of 5 rgb tuples

        for i in range(int(ribbons_per_emotion)):
            # choose one variant per-ribbon to ensure diversity
            c_rgb = variants[i % len(variants)]
            # slight random tweak
            tweak = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-8, 8)])
            c_rgb = tuple(np.clip(np.array(c_rgb) + tweak, 0, 255).astype(int).tolist())

            # random start
            x = rng.uniform(0, width-1)
            y = rng.uniform(0, height-1)
            pts = []
            ang_scale = 1.0 + curve_noise*rng.uniform(0.8, 1.2)

            for _ in range(int(steps)):
                ix = int(np.clip(x, 0, width-1))
                iy = int(np.clip(y, 0, height-1))
                a = angle[iy, ix] * ang_scale
                # step
                x += np.cos(a) * step_len
                y += np.sin(a) * step_len
                if x < -10 or x > width+10 or y < -10 or y > height+10:
                    break
                if len(pts) == 0 or (abs(pts[-1][0]-x) + abs(pts[-1][1]-y)) > 0.8:
                    pts.append((float(x), float(y)))

            if len(pts) >= 2:
                # slight width jitter
                wj = stroke_width + rng.integers(-1, 2)
                draw_polyline(canvas, pts, c_rgb, width=max(1, wj), alpha=ribbon_alpha, blur_px=stroke_blur)

    # composite
    base.alpha_composite(canvas)
    return base.convert("RGB")

# =========================
# Auto Brightness Compensation
# =========================
def auto_brightness_compensation(img_arr, target_mean=0.52, strength=0.9,
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
# UI — Sidebar
# =========================
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
        "A moment of awe as the sky shines with green light.",
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

# Auto-select top 3 emotions after filtering range
df_filtered = df[(df["compound"]>=cmp_min)&(df["compound"]<=cmp_max)]
counts = df_filtered["emotion"].value_counts()
top3 = counts.index.tolist()[:3] if len(counts)>0 else ["calm","awe","trust"]

# Build options (include customs)
custom_emotions = sorted(set(base_palette.keys()) - set(DEFAULT_RGB.keys()))
all_emotions_for_ui = list(ALL_EMOTIONS) + [e for e in custom_emotions if e not in ALL_EMOTIONS]

def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = base_palette.get(e, (0, 0, 0))
    return f"{e} (Custom {r},{g},{b})"

options_labels = [_label_emotion(e) for e in all_emotions_for_ui]
default_labels = [_label_emotion(e) for e in top3]
selected_labels = st.sidebar.multiselect("Selected Emotions (auto-picked 3; you can add/remove):", options_labels, default=default_labels)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

# Keep only rows with chosen emotions
df = df_filtered[df_filtered["emotion"].isin(selected_emotions)] if len(selected_emotions)>0 else df_filtered

# ---- 3) Ribbon Engine — Flow
st.sidebar.header("3) Ribbon Engine — Flow")
ribbons_per_emotion = st.sidebar.slider("Ribbons per Emotion", 2, 48, 16, 1)
stroke_width = st.sidebar.slider("Stroke Width", 1, 16, 5, 1)
steps = st.sidebar.slider("Ribbon Length (steps)", 120, 1400, 520, 10)
step_len = st.sidebar.slider("Step Length (px)", 0.5, 8.0, 2.6, 0.1)
curve_noise = st.sidebar.slider("Curve Randomness", 0.00, 0.80, 0.32, 0.01)
stroke_blur = st.sidebar.slider("Stroke Softness (blur px)", 0.0, 10.0, 0.6, 0.2)
ribbon_alpha = st.sidebar.slider("Ribbon Alpha", 60, 255, 230, 5)
bg_brightness = st.sidebar.slider("Background Brightness (dominant emotion)", 0.6, 1.8, 1.20, 0.05)

# ---- 4) Cinematic Color System (Controls)
st.sidebar.header("4) Cinematic Color System")
palette_mode = st.sidebar.selectbox("Palette Preset", list(CINEMATIC_PRESETS.keys()),
                                    index=list(CINEMATIC_PRESETS.keys()).index("Planetary (Soft)"))

exp = st.sidebar.slider("Exposure (stops)", -0.2, 1.8, 0.60, 0.01)
contrast = st.sidebar.slider("Contrast", 0.70, 1.90, 1.18, 0.01)
saturation = st.sidebar.slider("Saturation", 0.70, 2.00, 1.18, 0.01)
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40, 0.92, 0.01)
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50, 0.40, 0.01)

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature (Blue ↔ Red)", -1.0, 1.0, 0.00, 0.01)
tint = st.sidebar.slider("Tint (Green ↔ Magenta)", -1.0, 1.0, 0.02, 0.01)

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, 0.08, 0.01)
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, 0.06, 0.01)
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, 0.12, 0.01)
hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, 0.12, 0.01)
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, 0.10, 0.01)
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, 0.08, 0.01)
tone_balance = st.sidebar.slider("Tone Balance (Shadows ↔ Highlights)", -1.0, 1.0, 0.0, 0.01)

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius (px)", 0.0, 20.0, 7.0, 0.5)
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0, 0.42, 0.01)
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8, 0.16, 0.01)

# ---- 5) Auto Brightness Compensation
st.sidebar.header("5) Auto Brightness Compensation")
auto_bright = st.sidebar.checkbox("Enable Auto Brightness", value=True)
target_mean = st.sidebar.slider("Target Mean Luminance", 0.30, 0.75, 0.52, 0.01)
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, 0.90, 0.05)
abc_black = st.sidebar.slider("Black Point Percentile", 0.00, 0.20, 0.05, 0.01)
abc_white = st.sidebar.slider("White Point Percentile", 0.80, 1.00, 0.997, 0.001)
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0, 2.6, 0.05)

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

# ---- 7) Reset
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All"):
    st.session_state.clear()
    st.rerun()

# =========================
# DRAW SECTION
# =========================
left, right = st.columns([0.60,0.40])

with left:
    st.subheader("🎐 Ribbon Flow")

    # Palette preset
    working_palette, preset_temp, preset_tint = apply_palette_preset(get_active_palette(), palette_mode)

    # Render ribbons (seed varies each run)
    img = render_ribbons(
        df=df if not df.empty else pd.DataFrame({"emotion":["calm","awe","trust"]}),
        palette=working_palette,
        width=1500, height=850, seed=np.random.randint(0, 999999),
        emotions_in_use=selected_emotions,
        ribbons_per_emotion=ribbons_per_emotion, steps=steps, step_len=step_len,
        stroke_width=stroke_width, curve_noise=curve_noise,
        ribbon_alpha=ribbon_alpha, stroke_blur=stroke_blur,
        bg_brightness=bg_brightness
    )

    # ======== Cinematic Color Pipeline ========
    arr = np.array(img).astype(np.float32)/255.0

    # exposure (linear domain)
    lin = srgb_to_linear(arr)
    lin = lin * (2.0 ** exp)

    # white balance: preset + user
    lin = apply_white_balance(lin, temp + preset_temp, tint + preset_tint)

    # highlight roll-off (in linear)
    lin = highlight_rolloff(lin, roll)

    # back to display domain + filmic soft clamp
    arr = linear_to_srgb(np.clip(lin, 0, 4))
    arr = np.clip(filmic_tonemap(arr*1.25), 0, 1)

    # contrast, saturation, gamma
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    # split toning
    arr = split_tone(arr, (sh_r, sh_g, sh_b), (hi_r, hi_g, hi_b), tone_balance)

    # auto brightness compensation (AFTER creative grading)
    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    # bloom & vignette
    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)

    # ensure colorfulness
    arr = ensure_colorfulness(arr, min_sat=0.14, boost=1.16)

    final_img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="RGB")
    buf=BytesIO(); final_img.save(buf, format="PNG"); buf.seek(0)
    st.image(buf,use_column_width=True)
    st.download_button("💾 Download PNG",data=buf,file_name="ribbon_flow_final.png",mime="image/png")

with right:
    st.subheader("📊 Data & Emotion")
    df2=df.copy()
    if not df2.empty:
        df2["emotion_display"]=df2["emotion"].apply(lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})")
        cols=["text","emotion_display","compound","pos","neu","neg"]
        if "timestamp" in df2.columns: cols.insert(1,"timestamp")
        if "source" in df2.columns: cols.insert(2,"source")
        st.dataframe(df2[cols],use_container_width=True,height=600)
    else:
        st.info("No rows after filters. Try widening the compound range or different keyword.")

st.markdown("---")
st.caption("© 2025 Emotional Ribbon — Flow Edition (High-Brightness, Emotion-Driven Background, Top-3 Auto Select)")
