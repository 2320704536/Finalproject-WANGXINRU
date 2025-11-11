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
st.set_page_config(page_title="Emotional Aurora — Wang Xinru — Final Project", page_icon="🌌", layout="wide")
st.title("🌌 Emotional Aurora — Wang Xinru — Final Project")

# ✅ Instructions
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use

This project turns live news emotions into visuals with a **Corona-style aurora** or **Ribbon Flow** renderer, graded by a **cinematic color system** with **Auto Brightness Compensation** so images stay bright and rich.

1) **Fetch Data** – NewsAPI only. Enter a keyword (e.g., *aurora borealis*, *AI*, *technology*).
2) **Emotion Mapping** – VADER → curated emotions; filter by compound range; fixed color per emotion (customizable).
3) **Engine** – Choose **Renderer** (Aurora/ Ribbon), tweak swirl/ bands/ blur/ blend/ theme.
4) **Cinematic Color** – Exposure/ Contrast/ Gamma/ Saturation, White Balance, Split-toning, Bloom, Vignette.
5) **Auto Brightness** – Adaptive gain + black/white point mapping to keep images bright (not washed out).
6) **Palette** – Fixed per emotion; add custom RGB; import/export CSV.
7) **Download** – Save PNG.
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

THEMES = {
    "Deep Night": ((0.02, 0.03, 0.08), (0.0, 0.0, 0.0)),
    "Polar Twilight": ((0.06, 0.08, 0.16), (0.0, 0.0, 0.0)),
    "Dawn Haze": ((0.10, 0.08, 0.12), (0.0, 0.0, 0.0)),
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
# Blends
# =========================
def apply_band(base_rgb, band_color, alpha, mode):
    c = np.array(band_color).reshape(1,1,3)
    a = alpha[...,None]
    if mode=="Additive":
        return 1-(1-base_rgb)*(1-c*a)
    elif mode=="Linear Dodge":
        return np.clip(base_rgb + c*a,0,1)
    return base_rgb*(1-a)+c*a

# =========================
# Aurora: Corona layer
# =========================
def render_corona_layer(width,height,rng,color_rgb,swirl=1.0,detail=0.8,blur_px=6):
    X = np.linspace(-1,1,width)[None,:]
    Y = np.linspace(-1,1,height)[:,None]
    cx, cy = rng.uniform(-0.2,0.2), rng.uniform(-0.2,0.1)
    dx = X-cx; dy = Y-cy
    r = np.sqrt(dx*dx + dy*dy)+1e-6
    theta = np.arctan2(dy,dx)
    theta = theta + swirl * r * 2.0
    stripe = np.cos(theta*3.0) * np.exp(-r*1.8)
    noise = fbm_noise(height,width,rng,octaves=5,base_scale=int(80+60*detail))
    base = (0.55 + 0.45*noise) * np.clip(stripe,0,1)
    base /= (base.max()+1e-6)
    pil = Image.fromarray((base*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_px))
    alpha = np.array(pil).astype(np.float32)/255.
    rgb = np.zeros((height,width,3),dtype=np.float32)
    rgb[:,:,0]=color_rgb[0]; rgb[:,:,1]=color_rgb[1]; rgb[:,:,2]=color_rgb[2]
    return rgb, alpha

# =========================
# 🎬 Cinematic Color System
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

def jitter_emotion_color(rgb, emo_key, amount=0.05):
    rng = np.random.default_rng(abs(hash(emo_key)) % (2**32))
    jitter = (rng.random(3)-0.5)*2*amount
    col = np.clip(np.array(rgb)/255.0 + jitter, 0, 1)
    return tuple((col*255).astype(int).tolist())

# =========================
# Ribbon Flow Renderer
# =========================
def _aa_canvas(w, h, scale=2):
    return Image.new("RGBA", (w*scale, h*scale), (0,0,0,0)), scale

def _quad_bezier_points(p0, p1, p2, n=240):
    t = np.linspace(0,1,n)
    pts = (1-t)**2[:,None]*p0 + 2*(1-t)[:,None]*t[:,None]*p1 + t**2[:,None]*p2
    return pts

def render_ribbon_flow(df, palette, theme_name, width, height, seed,
                       bands, swirl, detail, blur, bg_brightness):
    """
    目标风格：蓝白丝带、干净留白、少量质感噪点。
    - 以主题色系(情绪色)生成多条丝带（粗细变化）
    - 背景为浅蓝（从 Theme 推出），叠加颗粒
    - 抗锯齿：放大绘制再缩回
    """
    rng = np.random.default_rng(seed)
    # 背景：从主题取顶部色，推到浅蓝
    top, bottom = THEMES[theme_name]
    top = np.array(top)*bg_brightness
    bg_rgb = np.clip(0.7*top + 0.3*np.array([0.75,0.86,0.94]), 0, 1)  # 淡蓝
    bg = Image.new("RGB", (width, height),
                   tuple((bg_rgb*255).astype(int).tolist()))

    # 超采样画布
    canvas, S = _aa_canvas(width, height, scale=2)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 可见情绪（没有就用 calm/hope/awe）
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions: emotions = ["calm","hope","awe"]

    total_bands = max(6, int(bands)*len(emotions))  # 更饱满
    # 丝带参数
    base_thick = int(22*S)
    jitter_thick = int(14*S)
    curve_amp = (0.25 + 0.5*swirl)  # 弯曲程度
    wiggle = (0.4 + 0.8*detail)     # 摆动

    for i in range(total_bands):
        emo = emotions[i % len(emotions)]
        base_col = np.array(palette.get(emo, (180,200,220)))/255.0
        # 颜色：以白色为主，搭配一两条深蓝强调线
        if i % 5 == 0:
            # 强调深线
            col = np.clip(0.35*base_col + 0.35*np.array([0.2,0.4,0.6]) + 0.3, 0, 1)
            alpha = 210
            width_px = max(4*S, int(0.35*base_thick + rng.integers(0, 8*S)))
        else:
            col = np.clip(0.75*base_col + 0.25*np.array([1.0,1.0,1.0]), 0, 1)
            alpha = 235
            width_px = base_thick + rng.integers(-jitter_thick, jitter_thick)

        # 随机三点的二次贝塞尔（左右跨越+上下起伏）
        x0 = -width*0.1; x2 = width*1.1
        y_mid = rng.uniform(height*0.2, height*0.8)
        y0 = y_mid + rng.uniform(-0.25,0.25)*height*wiggle
        y2 = y_mid + rng.uniform(-0.25,0.25)*height*wiggle
        ctrl_y = y_mid + rng.uniform(-0.35,0.35)*height*curve_amp
        p0 = np.array([x0, y0]); p1 = np.array([width*0.5, ctrl_y]); p2 = np.array([x2, y2])

        pts = _quad_bezier_points(p0, p1, p2, n=320)
        # 轻微扰动，避免太机械
        jitter = rng.normal(scale=0.6*S, size=pts.shape)
        pts += jitter

        # 画主线（粗线）
        rgb255 = tuple((col*255).astype(int).tolist())
        draw.line([tuple(p) for p in pts], fill=rgb255 + (alpha,), width=width_px)

        # 边缘高光（白色细线，提亮层次）
        edge_w = max(2*S, width_px//10)
        draw.line([tuple(p + np.array([0, -edge_w*0.6])) for p in pts],
                  fill=(255,255,255, min(180, alpha)), width=int(edge_w))
        draw.line([tuple(p + np.array([0,  edge_w*0.6])) for p in pts],
                  fill=(255,255,255, min(160, alpha)), width=int(edge_w*0.9))

        # 少量孔洞/分段（留白感）
        if rng.random() < 0.35:
            k0 = rng.integers(40, 220); k1 = k0 + rng.integers(12, 36)
            draw.line([tuple(p) for p in pts[k0:k1]],
                      fill=(0,0,0,0), width=width_px+int(6*S))

    # 颗粒质感（蓝白小点）
    grain = np.array(bg).astype(np.float32)/255.0
    gn = rng.random((height*2, width*2)).astype(np.float32)
    gn = (gn - 0.5)*0.08  # 控制幅度
    grain = np.clip(np.tile(grain,(2,2,1)) + gn[...,None], 0, 1)
    grain_img = Image.fromarray((grain*255).astype(np.uint8))

    # 合成：背景 + 丝带 + 模糊柔化
    if blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(blur)*0.8))
    comp = Image.new("RGB", (width, height), (0,0,0))
    comp.paste(grain_img, (0,0))
    comp = comp.convert("RGBA")
    comp.alpha_composite(canvas.resize((width, height), Image.LANCZOS))
    return comp.convert("RGB")

# =========================
# Aurora Engine (Corona)
# =========================
def render_engine_aurora(df,palette,theme_name,width,height,seed,blend_mode,bands,swirl,detail,blur,bg_brightness):
    rng = np.random.default_rng(seed)
    top,bottom = THEMES[theme_name]
    bg = vertical_gradient(width,height,top,bottom,brightness=bg_brightness)
    base = np.array(bg).astype(np.float32)/255.0
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions: emotions = ["hope","calm","awe"]
    for emo in emotions:
        raw_rgb = palette.get(emo, palette.get("mixed",(210,190,140)))
        emo_rgb = np.array(jitter_emotion_color(raw_rgb, emo, amount=0.04))/255.0
        for _ in range(max(1,int(bands))):
            _,alpha = render_corona_layer(width,height,rng,emo_rgb,swirl=swirl,detail=detail,blur_px=blur)
            base = apply_band(base, tuple(emo_rgb.tolist()), alpha, blend_mode)
    out = (np.clip(base,0,1)*255).astype(np.uint8)
    return Image.fromarray(out)

# =========================
# Auto Brightness Compensation
# =========================
def auto_brightness_compensation(img_arr, target_mean=0.42, strength=0.85,
                                 black_point_pct=0.06, white_point_pct=0.995,
                                 max_gain=2.2):
    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)
    Y = 0.2126*lin[:,:,0] + 0.7152*lin[:,:,1] + 0.0722*lin[:,:,2]
    bp = np.quantile(Y, black_point_pct); wp = np.quantile(Y, white_point_pct)
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
        "Calm atmospheric conditions create a beautiful environment."
