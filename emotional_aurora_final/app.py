import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt  # kept for consistency in env
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
    "joy":        (230, 200, 110),  # Warm Jupiter Gold
    "love":       (235, 180, 175),  # Venus Rose
    "pride":      (200, 170, 210),  # Saturn Violet
    "hope":       (160, 220, 200),  # Uranus Mint
    "curiosity":  (175, 210, 200),  # Soft Turquoise
    "calm":       (140, 180, 230),  # Neptune Blue
    "surprise":   (240, 190, 150),  # Dawn Peach
    "neutral":    (180, 180, 185),  # Lunar Gray
    "sadness":    (100, 130, 180),  # Deep Ocean Blue
    "anger":      (180, 80, 70),    # Mars Red
    "fear":       (130, 110, 160),  # Shadow Purple
    "disgust":    (130, 140, 110),  # Olive Gray
    "anxiety":    (210, 190, 140),  # Desert Sand
    "boredom":    (120, 120, 130),  # Slate Gray
    "nostalgia":  (235, 220, 190),  # Pale Cream
    "gratitude":  (175, 220, 220),  # Soft Cyan
    "awe":        (190, 230, 240),  # Ice Blue
    "trust":      (100, 170, 160),  # Sea Teal
    "confusion":  (210, 170, 175),  # Dust Pink
    "mixed":      (210, 190, 140),  # Pale Gold
}
ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy": "Warm Jupiter Gold","love": "Venus Rose","pride": "Saturn Violet","hope": "Uranus Mint",
    "curiosity": "Soft Turquoise","calm": "Neptune Blue","surprise": "Dawn Peach","neutral": "Lunar Gray",
    "sadness": "Deep Ocean Blue","anger": "Mars Red","fear": "Shadow Purple","disgust": "Olive Gray",
    "anxiety": "Desert Sand","boredom": "Slate Gray","nostalgia": "Pale Cream","gratitude": "Soft Cyan",
    "awe": "Ice Blue","trust": "Sea Teal","confusion": "Dust Pink","mixed": "Pale Gold",
}

# Background themes (vertical gradient for aurora sky)
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
# Background helper
# =========================

def vertical_gradient(width, height, top_rgb, bottom_rgb, brightness=1.0):
    # Ensure top and bottom are floats 0..1
    t = np.array(top_rgb, dtype=float) * brightness
    b = np.array(bottom_rgb, dtype=float) * brightness

    # Linear gradient from top → bottom
    grad = np.linspace(0, 1, height).reshape(height, 1, 1)

    # Broadcast to (height, 1, 3)
    img = t.reshape(1,1,3)*(1-grad) + b.reshape(1,1,3)*grad

    # Convert to uint8
    img = (img * 255).astype(np.uint8)

    # Expand horizontally to (height, width, 3)
    img = np.tile(img, (1, width, 1))

    # ✅ Ensure C-contiguous (PIL requirement)
    img = np.ascontiguousarray(img)

    return Image.fromarray(img, mode="RGB")

# =========================
# Aurora generator
# =========================
def make_wave_y(width, rng, style, amplitude_px, distortion, smoothness, base_y_px):
    x = np.linspace(0, 1, width)
    if style == "Smooth Sine":
        freq = rng.uniform(1.0, 3.0)
        phase = rng.uniform(0, 2*np.pi)
        y = np.sin(2*np.pi*freq*x + phase)
        noise = rng.normal(0, distortion, size=width)
        # smooth noise
        k = max(3, int(5 + smoothness*20))
        kernel = np.ones(k)/k
        noise = np.convolve(noise, kernel, mode="same")
        y = y*0.7 + noise*0.3
    elif style == "Cubic Spline":
        # pseudo spline: random anchors + moving average smoothing
        anchors = 8 + int(smoothness*10)
        pts_x = np.linspace(0,1,anchors)
        pts_y = rng.normal(0, 0.6, size=anchors)  # broader motion
        y = np.interp(x, pts_x, pts_y)
        k = max(5, int(15 + smoothness*40))
        kernel = np.ones(k)/k
        y = np.convolve(y, kernel, mode="same")
        y = y / (np.max(np.abs(y))+1e-6)
        y += rng.normal(0, distortion*0.5, size=width)
    else:  # "Flat Layers"
        y = rng.normal(0, distortion*0.3, size=width)
        k = max(7, int(25 + smoothness*60))
        kernel = np.ones(k)/k
        y = np.convolve(y, kernel, mode="same")
        y = y / (np.max(np.abs(y))+1e-6) * 0.5  # flatter

    y_px = base_y_px + amplitude_px * y
    return np.clip(y_px, 0, None)

def draw_band_to_mask(mask_img: Image.Image, y_top: np.ndarray, thickness_px: int):
    """Fill polygon between y_top and y_top+thickness."""
    w, h = mask_img.size
    y_bottom = np.clip(y_top + thickness_px, 0, h-1)
    xs = np.arange(w)
    poly = [(int(xs[i]), int(y_top[i])) for i in range(w)] + \
           [(int(xs[i]), int(y_bottom[w-1-i])) for i in range(w)]
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(poly, fill=255)

def apply_band(base_rgb: np.ndarray, band_color: tuple, band_alpha: np.ndarray, mode: str):
    """
    base_rgb: (H,W,3) float 0..1
    band_color: (3,) float 0..1
    band_alpha: (H,W) float 0..1
    mode: 'Normal' | 'Additive' | 'Linear Dodge'
    """
    c = np.array(band_color).reshape(1,1,3)
    a = band_alpha[...,None]
    if mode == "Normal":
        base_rgb = base_rgb*(1-a) + c*a
    elif mode == "Additive":
        # Screen-like soft additive
        base_rgb = 1 - (1 - base_rgb) * (1 - c * a)
    else:  # Linear Dodge (add then clip)
        base_rgb = np.clip(base_rgb + c * a, 0.0, 1.0)
    return base_rgb

def render_aurora(
    df: pd.DataFrame, active_palette: dict, theme_name: str,
    width: int, height: int, seed: int,
    wave_style: str, blend_mode: str,
    amplitude_scale: float, distortion: float, smoothness: float,
    opacity: float, blur_px: float,
    layers_per_emotion_max: int, band_thickness_ratio: float,
    bg_brightness: float, altitude_range: tuple[float, float]
):
    rng = np.random.default_rng(seed)

    # Background
    top, bottom = THEMES[theme_name]
    bg = vertical_gradient(width, height, top, bottom, brightness=bg_brightness)
    base = np.array(bg).astype(np.float32) / 255.0  # (H,W,3)

    # Emotion frequencies -> number of bands per emotion
    freq = df["emotion"].value_counts().to_dict()
    if not freq:
        return Image.fromarray((base*255).astype(np.uint8))

    max_f = max(freq.values())
    # Loop emotions by descending freq to prioritize strong bands on top
    for emo, f in sorted(freq.items(), key=lambda kv: -kv[1]):
        color_rgb = np.array(active_palette.get(emo, active_palette.get("mixed", (210,190,140))))/255.0
        norm = f / max_f if max_f > 0 else 0
        bands = max(1, int(round(norm * layers_per_emotion_max)))
        for _ in range(bands):
            # Altitude (relative height position)
            alt_min, alt_max = altitude_range
            base_y_px = int(height * rng.uniform(alt_min, alt_max))
            amplitude_px = int(height * (0.05 + 0.20 * amplitude_scale) * (0.7 + 0.6*rng.random()))
            thickness_px = max(3, int(amplitude_px * band_thickness_ratio))
            # Wave curve
            y_top = make_wave_y(width, rng, wave_style, amplitude_px, distortion, smoothness, base_y_px)
            # Band mask
            mask = Image.new("L", (width, height), 0)
            draw_band_to_mask(mask, y_top, thickness_px)
            if blur_px > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_px))
            band_alpha = (np.array(mask).astype(np.float32)/255.0) * opacity
            # Blend
            base = apply_band(base, tuple(color_rgb.tolist()), band_alpha, blend_mode)

    out = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)

# =========================
# Sidebar — Logical sections
# =========================
with st.expander("Instructions", expanded=False):
    st.markdown("""
**How to use**
1) **Data Source**: fetch texts via NewsAPI.  
2) **Emotion Mapping**: filter emotions and compound range.  
3) **Aurora Settings**: style, blend, amplitude, distortion, opacity, blur, layers, altitude.  
4) **Custom Palette (RGB)**: add emotions, import/export CSV.  
5) **Output**: download or reset.  
""")

# ---- 1) Data Source (NEWS ONLY)
st.sidebar.header("1) Data Source (NewsAPI only)")
kw = st.sidebar.text_input("Keyword (e.g., economy / technology / climate):", "technology")
news_btn = st.sidebar.button("Fetch from NewsAPI", use_container_width=True)

df = pd.DataFrame()
if news_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY. Add it in Streamlit Secrets.")
    else:
        df = fetch_news(key, keyword=kw)

# Fallback sample so app can render before first fetch
if df.empty:
    df = pd.DataFrame({"text":[
        "I can't believe how beautiful the sky is tonight!",
        "The new update is fantastic and smooth.",
        "Why is it raining again? Feeling a bit low.",
        "Our team finally shipped the feature! Proud and grateful.",
        "Markets look volatile; investors are anxious.",
    ]})
    df["timestamp"] = str(date.today())

# Ensure dataset has 'text'
if "text" not in df.columns:
    st.error("The dataset must include a 'text' column."); st.stop()

# Sentiment + emotion
with st.spinner("Analyzing sentiment and mapping emotions..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

# ---- 2) Emotion Mapping
st.sidebar.header("2) Emotion Mapping")
cmp_min, cmp_max = st.sidebar.slider("Compound range:", -1.0, 1.0, (-1.0, 1.0), 0.01)
available_emotions = sorted(df["emotion"].unique().tolist())

# Palette/init
init_palette_state()
ACTIVE_PALETTE = get_active_palette()

def emotion_label_with_name(e: str, pal: dict) -> str:
    if e in COLOR_NAMES: return f"{e} ({COLOR_NAMES[e]})"
    rgb = pal.get(e, (0,0,0)); return f"{e} (Custom {rgb[0]},{rgb[1]},{rgb[2]})"

final_labels_options = [emotion_label_with_name(e, ACTIVE_PALETTE) for e in ALL_EMOTIONS]
final_labels_default = [emotion_label_with_name(e, ACTIVE_PALETTE) for e in available_emotions]
selected_labels = st.sidebar.multiselect("Show emotions:", options=final_labels_options, default=final_labels_default)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

df = df[(df["emotion"].isin(selected_emotions)) & (df["compound"] >= cmp_min) & (df["compound"] <= cmp_max)].reset_index(drop=True)

# ---- 3) Aurora Settings (wave style & blend mode & controls)
st.sidebar.header("3) Aurora Settings")
wave_style = st.sidebar.selectbox("Wave Style:", ["Smooth Sine", "Cubic Spline", "Flat Layers"], index=1)  # default: Cubic Spline
blend_mode = st.sidebar.selectbox("Blend Mode:", ["Additive", "Linear Dodge", "Normal"], index=0)

amplitude_scale = st.sidebar.slider("Amplitude Scale:", 0.1, 2.0, 1.0, 0.05)
distortion = st.sidebar.slider("Distortion Strength:", 0.0, 1.5, 0.35, 0.05)
smoothness = st.sidebar.slider("Smoothness:", 0.0, 1.0, 0.6, 0.05)
opacity = st.sidebar.slider("Band Opacity:", 0.15, 1.0, 0.45, 0.05)
blur_px = st.sidebar.slider("Band Blur (px):", 0.0, 8.0, 2.0, 0.5)

layers_per_emotion_max = st.sidebar.slider("Max Layers per Emotion:", 1, 8, 4, 1)
band_thickness_ratio = st.sidebar.slider("Band Thickness (vs amplitude):", 0.1, 1.0, 0.45, 0.05)

theme_name = st.sidebar.selectbox("Background Theme:", list(THEMES.keys()), index=0)
bg_brightness = st.sidebar.slider("Background Brightness:", 0.3, 1.5, 1.0, 0.05)
altitude_range = st.sidebar.slider("Aurora Altitude Range (relative):", 0.0, 1.0, (0.25, 0.85), 0.01)

# ---- 4) Custom Palette (RGB)
st.sidebar.header("4) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette (RGB editor)", value=st.session_state["use_csv_palette"])
st.session_state["use_csv_palette"] = use_csv

with st.sidebar.expander("Add Custom Emotion (RGB 0–255)", expanded=False):
    c1, c2, c3, c4 = st.columns([1.8, 1, 1, 1])
    emo_name = c1.text_input("Emotion name")
    r = c2.number_input("R (0–255)", 0, 255, 210, 1)
    g = c3.number_input("G (0–255)", 0, 255, 190, 1)
    b = c4.number_input("B (0–255)", 0, 255, 140, 1)
    if st.button("Add Emotion", use_container_width=True):
        add_custom_emotion(emo_name, r, g, b)
        st.success(f"Added: {emo_name} = ({r},{g},{b})")
    custom_now = st.session_state.get("custom_palette", {})
    if custom_now:
        df_custom = pd.DataFrame([{"emotion": k, "r": v[0], "g": v[1], "b": v[2]} for k, v in sorted(custom_now.items())])
        st.dataframe(df_custom, use_container_width=True, height=200)
    else:
        st.caption("No custom colors yet.")

with st.sidebar.expander("Edit Palette / Import & Export CSV", expanded=False):
    upcsv = st.file_uploader("Import palette CSV (emotion,r,g,b)", type=["csv"])
    if upcsv is not None:
        import_palette_csv(upcsv)
    current_pal = dict(DEFAULT_RGB)
    current_pal.update(st.session_state.get("custom_palette", {}))
    if st.session_state.get("use_csv_palette"):
        current_pal = dict(st.session_state.get("custom_palette", {}))
    if current_pal:
        pal_df = pd.DataFrame([{"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in sorted(current_pal.items())])
        st.dataframe(pal_df, use_container_width=True, height=220)
        dl = export_palette_csv(current_pal)
        st.download_button("Download CSV", data=dl, file_name="palette.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No colors yet. Add emotions above or import a CSV.")

# ---- 5) Output
st.sidebar.header("5) Output")
if st.sidebar.button("Reset all settings"):
    st.session_state.clear(); st.rerun()

# =========================
# Draw & table
# =========================
left, right = st.columns([0.58, 0.42])
with left:
    st.subheader("🌌 Aurora")
    if df.empty:
        st.warning("No data points under current filters.")
    else:
        img = render_aurora(
            df=df, active_palette=ACTIVE_PALETTE, theme_name=theme_name,
            width=1600, height=900, seed=42,
            wave_style=wave_style, blend_mode=blend_mode,
            amplitude_scale=amplitude_scale, distortion=distortion, smoothness=smoothness,
            opacity=opacity, blur_px=blur_px,
            layers_per_emotion_max=layers_per_emotion_max, band_thickness_ratio=band_thickness_ratio,
            bg_brightness=bg_brightness, altitude_range=altitude_range
        )
        buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf, caption=f"Emotion Aurora — {theme_name} — {wave_style} / {blend_mode}", use_column_width=True)
        st.download_button("💾 Download PNG", data=buf, file_name="emotion_aurora.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    df_show = df.copy()

    def label_for_table(e):
        if e in COLOR_NAMES: return f"{e} ({COLOR_NAMES[e]})"
        rgb = ACTIVE_PALETTE.get(e, (0,0,0)); return f"{e} (Custom {rgb[0]},{rgb[1]},{rgb[2]})"

    df_show["emotion_label"] = df_show["emotion"].apply(label_for_table)
    cols = ["text", "emotion_label", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns: cols.insert(1, "timestamp")
    if "source" in df.columns: cols.insert(2, "source")
    st.dataframe(df_show[cols], use_container_width=True, height=520)

st.markdown("---")
st.caption("© 2025")
