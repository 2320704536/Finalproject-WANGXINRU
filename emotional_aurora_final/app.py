import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
import requests
from datetime import date
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ======================================================
# App Config
# ======================================================
st.set_page_config(
    page_title="Emotional Aurora — Wang Xinru — Final Project",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Emotional Aurora — Wang Xinru — Final Project")

# ======================================================
# Instructions
# ======================================================
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project

**1. Fetch Data from NewsAPI**  
Enter a keyword (examples below). Headlines will be analyzed.

**2. Sentiment → Emotion Mapping**  
Text is classified into 20+ curated emotional categories.

**3. Aurora Rendering (Corona Engine)**  
Each emotion generates aurora bands.  
Colors come from your palette + cinematic color engine.

**4. Custom Palette**  
Add your own RGB colors or import CSV palettes.

**5. Download**  
Save the final Aurora image as PNG.
---
""")

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
# Fetch NewsAPI
# ======================================================
def fetch_news(api_key, keyword="aurora", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()

        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "source": (a.get("source") or {}).get("name",""),
                "text": txt.strip(" -"),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error("Error fetching NewsAPI: " + str(e))
        return pd.DataFrame()

# ======================================================
# Default Emotion Colors
# ======================================================
DEFAULT_RGB = {
    "joy": (230,200,110),"love":(235,180,175),"pride":(200,170,210),
    "hope":(160,220,200),"curiosity":(175,210,200),"calm":(140,180,230),
    "surprise":(240,190,150),"neutral":(180,180,185),"sadness":(100,130,180),
    "anger":(180,80,70),"fear":(130,110,160),"disgust":(130,140,110),
    "anxiety":(210,190,140),"boredom":(120,120,130),"nostalgia":(235,220,190),
    "gratitude":(175,220,220),"awe":(190,230,240),"trust":(100,170,160),
    "confusion":(210,170,175),"mixed":(210,190,140),
}

COLOR_NAMES = {
    "joy": "Warm Jupiter Gold","love": "Venus Rose","pride": "Saturn Violet",
    "hope": "Uranus Mint","curiosity": "Soft Turquoise","calm": "Neptune Blue",
    "surprise": "Dawn Peach","neutral": "Lunar Gray","sadness": "Deep Ocean Blue",
    "anger": "Mars Red","fear": "Shadow Purple","disgust": "Olive Gray",
    "anxiety": "Desert Sand","boredom": "Slate Gray","nostalgia": "Pale Cream",
    "gratitude": "Soft Cyan","awe": "Ice Blue","trust": "Sea Teal",
    "confusion": "Dust Pink","mixed": "Pale Gold",
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())

# ======================================================
# Emotion Classification
# ======================================================
def analyze_sentiment(text):
    if not isinstance(text,str) or not text.strip():
        return {"neg":0,"neu":1,"pos":0,"compound":0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"],row["neu"],row["neg"],row["compound"]
    if comp>=0.7 and pos>0.5: return "joy"
    if comp>=0.55 and pos>0.45: return "love"
    if comp>=0.45 and pos>0.40: return "pride"
    if 0.25<=comp<0.45 and pos>0.30: return "hope"
    if 0.10<=comp<0.25 and neu>=0.5: return "calm"
    if 0.25<=comp<0.60 and neu<0.5: return "surprise"
    if comp<=-0.65 and neg>0.5: return "anger"
    if -0.65<comp<=-0.40 and neg>0.45: return "fear"
    if -0.40<comp<=-0.15 and neg>=0.35: return "sadness"
    if neg>0.5 and neu>0.3: return "anxiety"
    if neg>0.45 and pos<0.1: return "disgust"
    if neu>0.75 and abs(comp)<0.1: return "boredom"
    if pos>0.35 and neu>0.4 and 0<=comp<0.25: return "trust"
    if pos>0.30 and neu>0.35 and -0.05<=comp<=0.05: return "nostalgia"
    if pos>0.25 and neg>0.25: return "mixed"
    if pos>0.20 and neu>0.50 and comp>0.05: return "curiosity"
    if neu>0.6 and 0.05<=comp<=0.15: return "awe"
    return "neutral"

# ======================================================
# Palette State
# ======================================================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    if st.session_state["use_csv_palette"]:
        return dict(st.session_state["custom_palette"])
    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette",{}))
    return merged

def add_custom_emotion(name,r,g,b):
    if not name: return
    st.session_state["custom_palette"][name.strip()] = (int(r),int(g),int(b))

def import_palette_csv(file):
    try:
        dfc = pd.read_csv(file)
        need = {"emotion","r","g","b"}
        cols = {c.lower():c for c in dfc.columns}
        if not need.issubset(cols.keys()):
            st.error("CSV must include emotion,r,g,b columns")
            return
        pal = {}
        for _,row in dfc.iterrows():
            emo = str(row[cols["emotion"]]).strip()
            try:
                r=int(row[cols["r"]]); g=int(row[cols["g"]]); b=int(row[cols["b"]])
            except:
                continue
            pal[emo]=(r,g,b)
        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors.")
    except Exception as e:
        st.error("CSV import error: "+str(e))

def export_palette_csv(pal):
    buf = BytesIO()
    pd.DataFrame(
        [{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]
    ).to_csv(buf,index=False)
    buf.seek(0)
    return buf

# ======================================================
# Gradient Background
# ======================================================
THEMES = {
    "Deep Night": ((0.02,0.03,0.08),(0,0,0)),
    "Polar Twilight": ((0.06,0.08,0.16),(0,0,0)),
    "Dawn Haze": ((0.12,0.09,0.12),(0,0,0)),
}

def vertical_gradient(width,height,top_rgb,bottom_rgb,brightness=1.0):
    t=np.array(top_rgb)*brightness
    b=np.array(bottom_rgb)*brightness
    grad=np.linspace(0,1,height).reshape(height,1,1)
    img = t.reshape(1,1,3)*(1-grad)+b.reshape(1,1,3)*grad
    img=(img*255).astype(np.uint8)
    img=np.tile(img,(1,width,1))
    img=np.ascontiguousarray(img)
    return Image.fromarray(img)

# ======================================================
# fBm Noise
# ======================================================
def fbm_noise(h,w,rng,octaves=5,base_scale=96,persistence=0.55,lacunarity=2.0):
    acc=np.zeros((h,w),dtype=np.float32)
    amp=1.0
    scale=base_scale
    for _ in range(octaves):
        gh=max(1,h//max(1,scale))
        gw=max(1,w//max(1,scale))
        g=rng.random((gh,gw)).astype(np.float32)
        layer=np.array(
            Image.fromarray((g*255).astype(np.uint8)).resize((w,h),Image.BICUBIC),
            dtype=np.float32
        )/255.
        acc+=layer*amp
        amp*=persistence
        scale=max(1,int(scale/lacunarity))
    acc-=acc.min()
    if acc.max()>1e-6:
        acc/=acc.max()
    return acc

# ======================================================
# Auto Brightness Compensation System
# ======================================================
def auto_brightness_compensate(img_arr, target_luma=0.34, max_boost=1.9):
    luma = (
        img_arr[:,:,0]*0.299 +
        img_arr[:,:,1]*0.587 +
        img_arr[:,:,2]*0.114
    )
    cur=float(np.mean(luma))
    if cur>=target_luma:
        return img_arr
    boost=target_luma/(cur+1e-6)
    boost=min(boost,max_boost)
    img_arr=img_arr*boost
    img_arr=np.clip(img_arr,0,1)
    return img_arr
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
st.set_page_config(page_title="Emotional Aurora — Wang Xinru — Final Project", page_icon="🌌", layout="wide")
st.title("🌌 Emotional Aurora — Wang Xinru — Final Project")

# ✅ Instructions section
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

This project visualizes real-time emotions extracted from news articles as aurora patterns (Corona only).

**1) Fetch Data**
- Use **NewsAPI** to fetch headlines
- Enter a keyword (e.g., *AI*, *aurora*, *technology*, *science*)
- Sentiment is analyzed using **VADER**

**2) Emotion Classification**
- Each text is mapped to one of 20+ curated emotions
- Filter by emotions / compound score

**3) Aurora Rendering**
- **Corona-style** aurora only
- Each emotion generates one or more aurora bands
- Color comes from your palette (default + custom RGB/CSV)

**4) Auto Brightness Compensation (ABC)**
- Measures scene luminance and automatically adjusts exposure & gamma
- Optional saturation boost and soft local-contrast

**5) Output**
- Download the generated PNG
""")

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
# Default planet-like emotion colors
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

# Background gradient themes
THEMES = {
    "Deep Night": ((0.02, 0.03, 0.08), (0.0, 0.0, 0.0)),
    "Polar Twilight": ((0.06, 0.08, 0.16), (0.0, 0.0, 0.0)),
    "Dawn Haze": ((0.10, 0.08, 0.12), (0.0, 0.0, 0.0)),
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
# Gradient Background
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

# =========================
# fBm Noise
# =========================
def fbm_noise(h, w, rng, octaves=5, base_scale=96, persistence=0.55, lacunarity=2.0):
    acc = np.zeros((h,w), dtype=np.float32)
    amp = 1.0
    scale = base_scale
    for _ in range(octaves):
        gh = max(1, h//max(1,scale))
        gw = max(1, w//max(1,scale))
        g = rng.random((gh,gw)).astype(np.float32)
        layer = np.array(
            Image.fromarray((g*255).astype(np.uint8)).resize((w,h), Image.BICUBIC),
            dtype=np.float32
        )/255.
        acc += layer*amp
        amp *= persistence
        scale = max(1, int(scale/lacunarity))
    acc -= acc.min()
    if acc.max()>1e-6:
        acc /= acc.max()
    return acc

# =========================
# Blend
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
# ✅ Corona Renderer
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
# 🎬 Auto Brightness Compensation (ABC)
# =========================
def compute_luminance(img):  # img in 0..1, shape (H,W,3)
    # Rec. 709 luminance
    return 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]

def saturation_boost(img, sat=1.0):
    if abs(sat-1.0) < 1e-6: return img
    gray = compute_luminance(img)[...,None]
    return np.clip(gray + (img - gray)*sat, 0.0, 1.0)

def soft_local_contrast(img, amount=0.0, radius=25):
    if amount <= 0: return img
    # Unsharp mask style using Gaussian blur from PIL
    pil = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    blur = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    base = np.asarray(blur).astype(np.float32)/255.0
    enhanced = np.clip(img*(1+amount) - base*amount, 0.0, 1.0)
    return enhanced

def auto_brightness_compensation(img, target_mean=0.24, gain_limit=(0.6, 2.4),
                                 gamma_base=1.0, highlight_protect=0.15):
    """
    img: float32 0..1
    target_mean: desired luminance mean
    gain_limit: (min_gain, max_gain)
    gamma_base: base gamma correction (>0, around 1.0~1.2)
    highlight_protect: 0..1, strength of rolloff to protect highlights
    """
    lum = compute_luminance(img)
    cur = float(np.mean(lum) + 1e-6)
    gain = np.clip(target_mean / cur, gain_limit[0], gain_limit[1])

    # Gamma: darker scenes -> smaller denominator -> stronger lift
    gamma = max(0.2, gamma_base / (0.7 + 0.3 * (cur / target_mean)))
    out = np.clip((img ** gamma) * gain, 0.0, 1.0)

    # Simple filmic roll-off for highlights
    if highlight_protect > 0:
        t = highlight_protect
        out = (out * (1 + t*out)) / (out + t)  # Reinhard-like

    return np.clip(out, 0.0, 1.0)

# =========================
# ✅ Aurora Engine — Corona Only (returns float array)
# =========================
def render_engine(df,palette,theme_name,width,height,seed,blend_mode,bands,swirl,detail,blur,bg_brightness):
    rng = np.random.default_rng(seed)
    top,bottom = THEMES[theme_name]
    bg = vertical_gradient(width,height,top,bottom,brightness=bg_brightness)
    base = np.array(bg).astype(np.float32)/255.0

    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["hope","calm","awe"]

    for emo in emotions:
        color = np.array(palette.get(emo,palette["mixed"]))/255.0
        for _ in range(max(1,int(bands))):
            _, alpha = render_corona_layer(width,height,rng,color,swirl=swirl,detail=detail,blur_px=blur)
            base = apply_band(base,color,alpha,blend_mode)

    return np.clip(base, 0.0, 1.0)  # float array

# =========================
# UI
# =========================

# ---- 1) Data Source (NewsAPI only)
st.sidebar.header("1) Data Source (NewsAPI only)")
# Keyword example AFTER the label (not in placeholder)
st.sidebar.markdown("**Keyword** *(e.g., AI, aurora borealis, space weather, technology)*")
keyword = st.sidebar.text_input("", value="")

fetch_btn = st.sidebar.button("Fetch News")
df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(key,keyword)

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
palette = get_active_palette()

available_emotions = sorted(df["emotion"].unique().tolist())
custom_emotions = sorted(set(palette.keys()) - set(DEFAULT_RGB.keys()))
all_emotions_for_ui = list(ALL_EMOTIONS) + [e for e in custom_emotions if e not in ALL_EMOTIONS]

def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = palette.get(e, (0, 0, 0))
    return f"{e} (Custom {r},{g},{b})"

options_labels = [_label_emotion(e) for e in all_emotions_for_ui]
default_labels = [_label_emotion(e) for e in available_emotions] if available_emotions else options_labels
selected_labels = st.sidebar.multiselect("Show Emotions:", options_labels, default=default_labels)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]
df=df[(df["emotion"].isin(selected_emotions))&(df["compound"]>=cmp_min)&(df["compound"]<=cmp_max)]

# ---- 3) Aurora Engine — Corona
st.sidebar.header("3) Aurora Engine — Corona")
blend_mode = st.sidebar.selectbox("Blend Mode", ["Additive","Linear Dodge","Normal"],index=0)
bands = st.sidebar.slider("Bands per Emotion",1,5,3,1)

st.sidebar.subheader("Corona Settings")
swirl = st.sidebar.slider("Swirl Strength",0.0,1.5,0.9,0.05)
detail = st.sidebar.slider("Detail Level",0.2,1.2,0.8,0.05)
blur = st.sidebar.slider("Blur (px)",0.0,14.0,6.0,0.5)

theme_name = st.sidebar.selectbox("Background Theme", list(THEMES.keys()),index=0)
bg_brightness = st.sidebar.slider("Background Brightness",0.4,1.6,1.0,0.05)

# ---- 4) Color & ABC (NEW)
st.sidebar.header("4) Color & Auto-Brightness")
sat = st.sidebar.slider("Saturation", 0.8, 1.6, 1.15, 0.01)
abc_on = st.sidebar.checkbox("Enable Auto Brightness Compensation (ABC)", value=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    target_mean = st.slider("Target Luminance", 0.18, 0.38, 0.26, 0.01)
    gamma_base = st.slider("Base Gamma", 0.7, 1.4, 1.05, 0.01)
with col2:
    max_gain = st.slider("Max Gain", 1.2, 3.0, 2.2, 0.1)
    hl_protect = st.slider("Highlight Protect", 0.00, 0.40, 0.18, 0.01)
loc_contrast = st.sidebar.slider("Soft Local Contrast", 0.00, 0.60, 0.18, 0.01)

# ---- 5) Custom Palette
st.sidebar.header("5) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette",value=st.session_state["use_csv_palette"])
st.session_state["use_csv_palette"]=use_csv

with st.sidebar.expander("Add Custom Emotion",False):
    colA,colB,colC,colD=st.columns([1.8,1,1,1])
    name=colA.text_input("Emotion")
    r=colB.number_input("R",0,255,210)
    g=colC.number_input("G",0,255,190)
    b=colD.number_input("B",0,255,140)
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

# ---- 6) Reset
st.sidebar.header("6) Output")
if st.sidebar.button("Reset All"):
    st.session_state.clear()
    st.rerun()

# =========================
# DRAW SECTION
# =========================
left, right = st.columns([0.6,0.4])

with left:
    st.subheader("🌌 Aurora")

    if df.empty:
        st.warning("No data points under current filters.")
    else:
        img_arr = render_engine(
            df,palette,theme_name,
            width=1500,height=850,
            seed=np.random.randint(0,999999),
            blend_mode=blend_mode,
            bands=bands,
            swirl=swirl,
            detail=detail,
            blur=blur,
            bg_brightness=bg_brightness
        )

        # Color pipeline: saturation -> ABC -> local contrast
        img_arr = saturation_boost(img_arr, sat=sat)
        if abc_on:
            img_arr = auto_brightness_compensation(
                img_arr, target_mean=target_mean,
                gain_limit=(0.6, float(max_gain)),
                gamma_base=gamma_base,
                highlight_protect=hl_protect
            )
        img_arr = soft_local_contrast(img_arr, amount=loc_contrast, radius=24)

        out = (np.clip(img_arr,0,1)*255).astype(np.uint8)
        img = Image.fromarray(out)
        buf=BytesIO(); img.save(buf, format="PNG"); buf.seek(0)

        st.image(buf,use_column_width=True)
        st.download_button("💾 Download PNG",data=buf,file_name="aurora_corona.png",mime="image/png")

with right:
    st.subheader("📊 Data & Emotion")
    df2=df.copy()
    df2["emotion_display"]=df2["emotion"].apply(
        lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})"
    )
    cols=["text","emotion_display","compound","pos","neu","neg"]
    if "timestamp" in df.columns: cols.insert(1,"timestamp")
    if "source" in df.columns: cols.insert(2,"source")
    st.dataframe(df2[cols],use_container_width=True,height=600)

st.markdown("---")
st.caption("© 2025 Emotional Aurora — Corona Edition (with Auto Brightness Compensation)")
