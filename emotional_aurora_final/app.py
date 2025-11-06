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
def fbm_noise(h, w, rng, octaves=5, base_scale=96, persistence=0.55, lacunarity=2.0):
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
# Blend helper
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

# =========================
# Aurora — Arcs (帘幕)
# =========================
def render_arcs_layer(width, height, rng, color_rgb, strength=0.9, streak=0.7, blur_px=5.0):
    X = np.arange(width)[None, :].astype(np.float32)
    Y = np.arange(height)[:, None].astype(np.float32)
    base = np.zeros((height, width), dtype=np.float32)

    # 基础水平弧带（下部更亮）
    y0 = rng.uniform(0.55, 0.75) * height
    sigma_y = rng.uniform(0.10, 0.20) * height
    band = np.exp(-((Y - y0)**2) / (2.0 * sigma_y**2))
    base += band

    # 竖直光丝：列噪声 + 强度衰减
    col_noise = fbm_noise(1, width, rng, octaves=4, base_scale=48).reshape(1, width)
    streaks = np.repeat(col_noise, height, axis=0)
    streaks = (streaks - streaks.min()) / (streaks.max() - streaks.min() + 1e-6)
    streaks = streaks**2.0  # 更尖锐
    base *= (0.6 + 0.4 * streaks * streak)

    # 垂直亮度：自下而上逐渐衰减 + 上部轻微紫光过渡（不着色，仅亮度控制）
    vertical_profile = np.linspace(1.0, 0.25, height, dtype=np.float32).reshape(height,1)
    base *= vertical_profile

    # 平滑 + 模糊
    base = base / (base.max() + 1e-6)
    band_img = (np.clip(base * strength, 0.0, 1.0) * 255).astype(np.uint8)
    pil = Image.fromarray(band_img, mode="L")
    if blur_px > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_px))
    alpha = np.array(pil).astype(np.float32) / 255.0

    rgb = np.zeros((height,width,3), dtype=np.float32)
    rgb[:,:,0] = color_rgb[0]; rgb[:,:,1] = color_rgb[1]; rgb[:,:,2] = color_rgb[2]
    return rgb, alpha

# =========================
# Aurora — Corona (旋涡)
# =========================
def render_corona_layer(width, height, rng, color_rgb, swirl=0.9, detail=0.6, blur_px=6.0):
    X = np.linspace(-1, 1, width)[None, :]
    Y = np.linspace(-1, 1, height)[:, None]
    cx, cy = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.1)  # 近天顶
    dx = X - cx; dy = Y - cy
    r = np.sqrt(dx*dx + dy*dy) + 1e-6
    theta = np.arctan2(dy, dx)

    # 旋涡 + 半径衰减
    theta = theta + swirl * r * 2.0
    # 将角度扰动映射回位置，得到条纹
    stripe = np.cos(theta*3.0) * np.exp(-r*1.8)
    # 噪声细节
    noise = fbm_noise(height, width, rng, octaves=5, base_scale=int(80 + 60*detail), persistence=0.6)
    base = (0.55 + 0.45*noise) * np.clip(stripe, 0, 1)

    # 平滑 + 模糊
    base = base / (base.max() + 1e-6)
    band_img = (np.clip(base, 0.0, 1.0) * 255).astype(np.uint8)
    pil = Image.fromarray(band_img, mode="L").filter(ImageFilter.GaussianBlur(radius=blur_px))
    alpha = np.array(pil).astype(np.float32) / 255.0

    rgb = np.zeros((height,width,3), dtype=np.float32)
    rgb[:,:,0] = color_rgb[0]; rgb[:,:,1] = color_rgb[1]; rgb[:,:,2] = color_rgb[2]
    return rgb, alpha

# =========================
# Aurora Engine
# =========================
def render_engine(
    df: pd.DataFrame, palette: dict, theme_name: str,
    width: int, height: int, seed: int,
    aurora_type: str, preset: str,
    blend_mode: str, bands_per_emotion: int,
    arcs_strength: float, arcs_streak: float, arcs_blur: float,
    corona_swirl: float, corona_detail: float, corona_blur: float,
    global_brightness: float
):
    rng = np.random.default_rng(seed)
    top, bottom = THEMES[theme_name]
    bg = vertical_gradient(width, height, top, bottom, brightness=global_brightness)
    base = np.array(bg).astype(np.float32) / 255.0

    # 预设：控制整体观感
    if preset == "Realistic":
        blend_mode = "Additive"
        arcs_strength = min(arcs_strength, 0.9)
        arcs_streak = min(arcs_streak, 0.8)
        corona_detail = 0.5
    elif preset == "Artistic":
        blend_mode = "Linear Dodge"
        arcs_strength = max(arcs_strength, 0.95)
        corona_detail = 0.8
    else:  # Cinematic
        blend_mode = blend_mode  # 尊重用户选择
        arcs_strength = max(arcs_strength, 0.95)
        corona_detail = max(corona_detail, 0.7)

    # 情绪频率（决定出现优先级）
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["hope","calm","awe"]  # 保底美观

    types = ["Arcs","Corona","Mix"]
    if aurora_type == "Random":
        choose = rng.choice(types)
    else:
        choose = aurora_type

    for emo in emotions:
        color = np.array(palette.get(emo, palette.get("mixed",(210,190,140))))/255.0
        n = max(1, int(bands_per_emotion))
        for _ in range(n):
            if choose == "Arcs":
                rgb, alpha = render_arcs_layer(width,height,rng,color,
                                               strength=arcs_strength, streak=arcs_streak, blur_px=arcs_blur)
                base = apply_band(base, tuple(color.tolist()), alpha, blend_mode)
            elif choose == "Corona":
                rgb, alpha = render_corona_layer(width,height,rng,color,
                                                 swirl=corona_swirl, detail=corona_detail, blur_px=corona_blur)
                base = apply_band(base, tuple(color.tolist()), alpha, blend_mode)
            else:  # Mix
                if rng.random() < 0.6:
                    rgb, alpha = render_arcs_layer(width,height,rng,color,
                                                   strength=arcs_strength, streak=arcs_streak, blur_px=arcs_blur)
                else:
                    rgb, alpha = render_corona_layer(width,height,rng,color,
                                                     swirl=corona_swirl, detail=corona_detail, blur_px=corona_blur)
                base = apply_band(base, tuple(color.tolist()), alpha, blend_mode)

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
3) Aurora Engine (Type / Preset / Blend)  
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
        "The northern sky erupted with a breathtaking aurora tonight.",
        "New policies spark hope and gratitude among citizens.",
        "Markets look volatile; investors are anxious.",
        "Calm seas and clear weather bring peace to travelers.",
        "Innovation in science fills people with awe and curiosity."
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

# ---- 3) Aurora Engine (Cinematic preset)
st.sidebar.header("3) Aurora Engine")
aurora_type = st.sidebar.selectbox("Aurora Type:", ["Random","Arcs","Corona","Mix"], index=0)
preset = st.sidebar.selectbox("Visual Preset:", ["Realistic","Artistic","Cinematic"], index=2)  # default Cinematic
blend_mode = st.sidebar.selectbox("Blend Mode:", ["Additive","Linear Dodge","Normal"], index=0)

bands_per_emotion = st.sidebar.slider("Bands per Emotion:", 1, 5, 3, 1)

st.markdown("")  # spacing

with st.sidebar.expander("Arcs Settings", expanded=True if aurora_type in ["Random","Arcs","Mix"] else False):
    arcs_strength = st.slider("Brightness (Arcs):", 0.6, 1.2, 1.0, 0.05)
    arcs_streak = st.slider("Streak Strength:", 0.0, 1.2, 0.9, 0.05)
    arcs_blur = st.slider("Blur (px):", 0.0, 12.0, 5.0, 0.5)

with st.sidebar.expander("Corona Settings", expanded=True if aurora_type in ["Random","Corona","Mix"] else False):
    corona_swirl = st.slider("Swirl Strength:", 0.0, 1.5, 0.9, 0.05)
    corona_detail = st.slider("Detail Level:", 0.2, 1.2, 0.8, 0.05)
    corona_blur = st.slider("Blur (px):", 0.0, 14.0, 6.0, 0.5)

theme_name = st.sidebar.selectbox("Background Theme:", list(THEMES.keys()), index=0)
global_brightness = st.sidebar.slider("Background Brightness:", 0.4, 1.6, 1.0, 0.05)

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
    st.subheader("🎬 Cinematic Aurora (Randomized Corona / Arcs / Mix)")
    if df.empty:
        st.warning("No data points under current filters.")
    else:
        img = render_engine(
            df=df, palette=ACTIVE_PALETTE, theme_name=theme_name,
            width=1600, height=900, seed=np.random.randint(0, 10_000),
            aurora_type=aurora_type, preset=preset,
            blend_mode=blend_mode, bands_per_emotion=bands_per_emotion,
            arcs_strength=arcs_strength, arcs_streak=arcs_streak, arcs_blur=arcs_blur,
            corona_swirl=corona_swirl, corona_detail=corona_detail, corona_blur=corona_blur,
            global_brightness=global_brightness
        )
        buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf, caption=f"{aurora_type} — {preset} — {blend_mode}", use_column_width=True)
        st.download_button("💾 Download PNG", data=buf, file_name="emotion_aurora_cinematic.png", mime="image/png")

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
