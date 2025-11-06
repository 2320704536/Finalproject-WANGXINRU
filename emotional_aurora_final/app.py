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
st.set_page_config(page_title="Emotional Aurora — Wang Xinru — Final Project", page_icon="🌈", layout="wide")
st.title("🌈 Emotional Aurora — Wang Xinru — Final Project")

# =========================
# Resources
# =========================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()
sia = load_vader()

# =========================
# NewsAPI
# =========================
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {"q": keyword, "language": "en", "sortBy": "publishedAt",
              "pageSize": page_size, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        rows = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            desc = a.get("description") or ""
            txt = (title + " - " + desc).strip(" -")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt,
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# =========================
# Planetary Palette (soft, natural)
# =========================
DEFAULT_RGB = {
    "joy":        (230, 200, 110),
    "love":       (235, 180, 175),
    "pride":      (200, 170, 210),
    "hope":       (160, 220, 200),
    "curiosity":  (175, 210, 200),
    "calm":       (140, 180, 230),
    "surprise":   (240, 190, 150),
    "neutral":    (180, 180, 185),
    "sadness":    (100, 130, 180),
    "anger":      (180, 80, 70),
    "fear":       (130, 110, 160),
    "disgust":    (130, 140, 110),
    "anxiety":    (210, 190, 140),
    "boredom":    (120, 120, 130),
    "nostalgia":  (235, 220, 190),
    "gratitude":  (175, 220, 220),
    "awe":        (190, 230, 240),
    "trust":      (100, 170, 160),
    "confusion":  (210, 170, 175),
    "mixed":      (210, 190, 140),
}
ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy": "Warm Jupiter Gold","love": "Venus Rose","pride": "Saturn Violet","hope": "Uranus Mint",
    "curiosity": "Soft Turquoise","calm": "Neptune Blue","surprise": "Dawn Peach","neutral": "Lunar Gray",
    "sadness": "Deep Ocean Blue","anger": "Mars Red","fear": "Shadow Purple","disgust": "Olive Gray",
    "anxiety": "Desert Sand","boredom": "Slate Gray","nostalgia": "Pale Cream","gratitude": "Soft Cyan",
    "awe": "Ice Blue","trust": "Sea Teal","confusion": "Dust Pink","mixed": "Pale Gold",
}

# Background themes (top/bottom RGB floats)
THEMES = {
    "Deep Night": ((0.02, 0.03, 0.08), (0.00, 0.00, 0.00)),
    "Polar Twilight": ((0.06, 0.08, 0.16), (0.00, 0.00, 0.00)),
    "Dawn Haze": ((0.10, 0.08, 0.12), (0.00, 0.00, 0.00)),
}

# =========================
# Sentiment & emotion mapping
# =========================
def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row) -> str:
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]
    if comp >= 0.7 and pos > 0.5:             return "joy"
    if comp >= 0.55 and pos > 0.45:           return "love"
    if comp >= 0.45 and pos > 0.40:           return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30:    return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5:    return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5:     return "surprise"
    if comp <= -0.65 and neg > 0.5:           return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45:  return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3:               return "anxiety"
    if neg > 0.45 and pos < 0.1:              return "disgust"
    if neu > 0.75 and abs(comp) < 0.1:        return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25:             return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15:    return "awe"
    return "neutral"

# =========================
# Palette state (CSV + custom)
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state: st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state: st.session_state["custom_palette"] = {}

def get_active_palette():
    if st.session_state.get("use_csv_palette") and st.session_state.get("custom_palette"):
        return dict(st.session_state["custom_palette"])
    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged

def add_custom_emotion(emotion: str, r: int, g: int, b: int):
    if not emotion:
        st.warning("Emotion name cannot be empty."); return
    r = int(np.clip(r, 0, 255)); g = int(np.clip(g, 0, 255)); b = int(np.clip(b, 0, 255))
    st.session_state["custom_palette"][emotion.strip()] = (r, g, b)

def import_palette_csv(file):
    try:
        dfc = pd.read_csv(file)
        cols_lower = [c.lower() for c in dfc.columns]
        needed = {"emotion","r","g","b"}
        if not needed.issubset(set(cols_lower)):
            st.error("CSV must include columns: emotion, r, g, b"); return
        colmap = {c.lower(): c for c in dfc.columns}
        em = colmap["emotion"]; rc = colmap["r"]; gc = colmap["g"]; bc = colmap["b"]
        pal = {}
        for _, row in dfc.iterrows():
            emo = str(row[em]).strip()
            try:
                r = int(row[rc]); g = int(row[gc]); b = int(row[bc])
            except Exception:
                continue
            r = int(np.clip(r,0,255)); g = int(np.clip(g,0,255)); b = int(np.clip(b,0,255))
            if emo: pal[emo] = (r,g,b)
        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV.")
    except Exception as e:
        st.error(f"Failed to import CSV: {e}")

def export_palette_csv(palette_dict: dict) -> BytesIO:
    dfp = pd.DataFrame([{"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in palette_dict.items()])
    buf = BytesIO(); dfp.to_csv(buf, index=False); buf.seek(0); return buf

# =========================
# Background helper (robust)
# =========================
def vertical_gradient(width, height, top_rgb, bottom_rgb, brightness=1.0):
    t = np.array(top_rgb, dtype=float) * brightness
    b = np.array(bottom_rgb, dtype=float) * brightness
    grad = np.linspace(0, 1, height).reshape(height, 1, 1)
    img = t.reshape(1,1,3)*(1-grad) + b.reshape(1,1,3)*grad
    img = (img * 255).astype(np.uint8)
    img = np.tile(img, (1, width, 1))
    img = np.ascontiguousarray(img)
    return Image.fromarray(img, mode="RGB")

# =========================
# Perlin-like fBm Noise (no extra deps)
# =========================
def fbm_noise(h, w, rng, octaves=4, base_scale=128, persistence=0.5, lacunarity=2.0):
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    scale = base_scale
    for _ in range(octaves):
        gh = max(1, h // max(1, scale))
        gw = max(1, w // max(1, scale))
        g = rng.random((gh, gw)).astype(np.float32)
        layer = np.array(
            Image.fromarray((g*255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC),
            dtype=np.float32
        ) / 255.0
        acc += layer * amp
        amp *= persistence
        scale = max(1, int(scale / lacunarity))
    acc = acc - acc.min()
    if acc.max() > 1e-6:
        acc = acc / acc.max()
    return acc  # 0..1

# =========================
# Aurora renderer (REAL AURORA)
# =========================
def apply_band(base_rgb: np.ndarray, band_color: tuple, band_alpha: np.ndarray, mode: str):
    c = np.array(band_color).reshape(1,1,3)
    a = band_alpha[...,None]
    if mode == "Normal":
        base_rgb = base_rgb*(1-a) + c*a
    elif mode == "Additive":
        base_rgb = 1 - (1 - base_rgb) * (1 - c * a)
    else:  # Linear Dodge
        base_rgb = np.clip(base_rgb + c * a, 0.0, 1.0)
    return base_rgb

def render_aurora_real(
    df: pd.DataFrame, active_palette: dict, theme_name: str,
    width: int, height: int, seed: int,
    direction: str, blend_mode: str,
    base_width_ratio: float, noise_strength: float,
    opacity: float, blur_px: float,
    bands_per_emotion: int, bg_brightness: float
):
    rng = np.random.default_rng(seed)

    # Background
    top, bottom = THEMES[theme_name]
    bg = vertical_gradient(width, height, top, bottom, brightness=bg_brightness)
    base = np.array(bg).astype(np.float32) / 255.0

    # Global noise fields
    noise_main = fbm_noise(height, width, rng, octaves=5, base_scale=96, persistence=0.55, lacunarity=2.0)
    noise_line = fbm_noise(height, 1, rng, octaves=4, base_scale=64, persistence=0.6, lacunarity=2.0).reshape(height)  # for center meander

    # Emotion frequencies
    freq = df["emotion"].value_counts().to_dict()
    if not freq:
        return Image.fromarray((base*255).astype(np.uint8))

    max_f = max(freq.values())

    # Grid for vectorization
    X = np.arange(width)[None, :].astype(np.float32)  # (1,W)
    Y = np.arange(height)[:, None].astype(np.float32) # (H,1)

    # Direction params
    if direction == "Vertical Curtains":
        slope = 0.0
    elif direction == "Tilted Curtains":
        slope = rng.uniform(-0.25, 0.25)  # tilt amount
    else:  # "Horizontal Bands"
        slope = None  # special handling

    for emo, f in sorted(freq.items(), key=lambda kv: -kv[1]):
        color_rgb = np.array(active_palette.get(emo, active_palette.get("mixed", (210,190,140))))/255.0

        # number of bands per emotion (fixed=3 as requested)
        bands = int(bands_per_emotion)

        for _ in range(bands):
            # Width (sigma) as ratio of image width
            sigma = (base_width_ratio * width) * (0.6 + 0.6*rng.random())
            sigma = max(12.0, float(sigma))

            # Center curve
            cx0 = rng.uniform(0.2, 0.8) * width
            meander = (noise_line - 0.5) * (noise_strength * width * 0.15)
            center_x = cx0 + meander  # (H,)

            if slope is not None:
                # Add tilt: linear with Y
                center_x = center_x + slope * (Y.squeeze() - height/2.0)

            # Build alpha mask by Gaussian from center_x
            dx = X - center_x[:, None]  # (H,W)
            band = np.exp(-(dx*dx) / (2.0 * (sigma**2)))

            # Vertical brightness profile: brighter at top, fade to bottom
            vertical_profile = np.linspace(1.0, 0.35, height, dtype=np.float32).reshape(height,1)
            band *= vertical_profile

            # Modulate with noise texture to create aurora ripples
            ripple = 0.6 + 0.4 * (noise_main**1.2)
            band = band * ripple

            # Normalize and apply opacity
            band = band / (band.max() + 1e-6)
            band = band * opacity

            # Optional blur (as image)
            band_img = (np.clip(band, 0.0, 1.0) * 255).astype(np.uint8)
            band_pil = Image.fromarray(band_img, mode="L")
            if blur_px > 0:
                band_pil = band_pil.filter(ImageFilter.GaussianBlur(radius=blur_px))
            band_alpha = np.array(band_pil).astype(np.float32) / 255.0

            base = apply_band(base, tuple(color_rgb.tolist()), band_alpha, blend_mode)

    out = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)

# =========================
# UI
# =========================
with st.expander("Instructions", expanded=False):
    st.markdown("""
**How to use**
1) Data Source (NewsAPI)  
2) Emotion Mapping  
3) Aurora Settings (REAL Aurora)  
4) Custom Palette (RGB / CSV)  
5) Output  
""")

# ---- 1) Data Source (NEWS ONLY)
st.sidebar.header("1) Data Source (NewsAPI only)")
kw = st.sidebar.text_input("Keyword:", "technology")
news_btn = st.sidebar.button("Fetch from NewsAPI", use_container_width=True)

df = pd.DataFrame()
if news_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY. Add it in Streamlit Secrets.")
    else:
        df = fetch_news(key, keyword=kw)

# Fallback sample
if df.empty:
    df = pd.DataFrame({"text":[
        "I can't believe how beautiful the sky is tonight!",
        "The new update is fantastic and smooth.",
        "Why is it raining again? Feeling a bit low.",
        "Our team finally shipped the feature! Proud and grateful.",
        "Markets look volatile; investors are anxious.",
    ]})
    df["timestamp"] = str(date.today())

if "text" not in df.columns:
    st.error("Dataset must include 'text' column."); st.stop()

# Sentiment + emotion
with st.spinner("Analyzing sentiment and mapping emotions..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

# ---- 2) Emotion Mapping
st.sidebar.header("2) Emotion Mapping")
cmp_min, cmp_max = st.sidebar.slider("Compound range:", -1.0, 1.0, (-1.0, 1.0), 0.01)
available_emotions = sorted(df["emotion"].unique().tolist())

init_palette_state()
ACTIVE_PALETTE = get_active_palette()

def emotion_label_with_name(e: str, pal: dict) -> str:
    if e in COLOR_NAMES: return f"{e} ({COLOR_NAMES[e]})"
    r,g,b = pal.get(e, (0,0,0)); return f"{e} (Custom {r},{g},{b})"

final_labels_options = [emotion_label_with_name(e, ACTIVE_PALETTE) for e in ALL_EMOTIONS]
final_labels_default = [emotion_label_with_name(e, ACTIVE_PALETTE) for e in available_emotions]
selected_labels = st.sidebar.multiselect("Show emotions:", options=final_labels_options, default=final_labels_default)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

df = df[(df["emotion"].isin(selected_emotions)) & (df["compound"] >= cmp_min) & (df["compound"] <= cmp_max)].reset_index(drop=True)

# ---- 3) Aurora Settings (REAL)
st.sidebar.header("3) Aurora Settings (REAL)")
direction = st.sidebar.selectbox("Direction:", ["Vertical Curtains","Tilted Curtains","Horizontal Bands"], index=0)
blend_mode = st.sidebar.selectbox("Blend Mode:", ["Additive","Linear Dodge","Normal"], index=0)

base_width_ratio = st.sidebar.slider("Base Width (ratio of width):", 0.02, 0.20, 0.06, 0.01)
noise_strength = st.sidebar.slider("Meander / Noise Strength:", 0.0, 1.0, 0.45, 0.05)
opacity = st.sidebar.slider("Band Opacity:", 0.15, 0.95, 0.50, 0.05)
blur_px = st.sidebar.slider("Vertical Blur (px):", 0.0, 12.0, 4.5, 0.5)

# your choice: per emotion 3 bands (fixed)
bands_per_emotion = 3

theme_name = st.sidebar.selectbox("Background Theme:", list(THEMES.keys()), index=0)
bg_brightness = st.sidebar.slider("Background Brightness:", 0.4, 1.6, 1.0, 0.05)

# ---- 4) Custom Palette
st.sidebar.header("4) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette (RGB editor)", value=st.session_state["use_csv_palette"])
st.session_state["use_csv_palette"] = use_csv

with st.sidebar.expander("Add Custom Emotion (RGB 0–255)", expanded=False):
    c1,c2,c3,c4 = st.columns([1.8,1,1,1])
    emo_name = c1.text_input("Emotion name")
    r = c2.number_input("R (0–255)", 0, 255, 210, 1)
    g = c3.number_input("G (0–255)", 0, 255, 190, 1)
    b = c4.number_input("B (0–255)", 0, 255, 140, 1)
    if st.button("Add Emotion", use_container_width=True):
        add_custom_emotion(emo_name, r, g, b); st.success(f"Added: {emo_name} = ({r},{g},{b})")
    custom_now = st.session_state.get("custom_palette", {})
    if custom_now:
        st.dataframe(pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in sorted(custom_now.items())]),
                     use_container_width=True, height=180)
    else:
        st.caption("No custom colors yet.")

with st.sidebar.expander("Edit Palette / Import & Export CSV", expanded=False):
    upcsv = st.file_uploader("Import palette CSV (emotion,r,g,b)", type=["csv"])
    if upcsv is not None: import_palette_csv(upcsv)
    current_pal = dict(DEFAULT_RGB); current_pal.update(st.session_state.get("custom_palette", {}))
    if st.session_state.get("use_csv_palette"):
        current_pal = dict(st.session_state.get("custom_palette", {}))
    if current_pal:
        pal_df = pd.DataFrame([{"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in sorted(current_pal.items())])
        st.dataframe(pal_df, use_container_width=True, height=210)
        dl = export_palette_csv(current_pal)
        st.download_button("Download CSV", data=dl, file_name="palette.csv", mime="text/csv", use_container_width=True)

# ---- 5) Output
st.sidebar.header("5) Output")
if st.sidebar.button("Reset all settings"):
    st.session_state.clear(); st.rerun()

# =========================
# Draw & table
# =========================
left, right = st.columns([0.58, 0.42])
with left:
    st.subheader("🌌 Real Aurora")
    if df.empty:
        st.warning("No data points under current filters.")
    else:
        img = render_aurora_real(
            df=df, active_palette=ACTIVE_PALETTE, theme_name=theme_name,
            width=1600, height=900, seed=42,
            direction=direction, blend_mode=blend_mode,
            base_width_ratio=base_width_ratio, noise_strength=noise_strength,
            opacity=opacity, blur_px=blur_px,
            bands_per_emotion=bands_per_emotion, bg_brightness=bg_brightness
        )
        buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf, caption=f"Real Aurora — {direction} — {blend_mode}", use_column_width=True)
        st.download_button("💾 Download PNG", data=buf, file_name="emotion_aurora_real.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    df_show = df.copy()
    def label_for_table(e):
        if e in COLOR_NAMES: return f"{e} ({COLOR_NAMES[e]})"
        r,g,b = ACTIVE_PALETTE.get(e, (0,0,0)); return f"{e} (Custom {r},{g},{b})"
    df_show["emotion_label"] = df_show["emotion"].apply(label_for_table)
    cols = ["text", "emotion_label", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns: cols.insert(1, "timestamp")
    if "source" in df.columns: cols.insert(2, "source")
    st.dataframe(df_show[cols], use_container_width=True, height=520)

st.markdown("---")
st.caption("© 2025")
