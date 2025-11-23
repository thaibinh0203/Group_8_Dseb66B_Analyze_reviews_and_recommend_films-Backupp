import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from base64 import b64encode
import numpy as np
from collections import Counter
import os

# ===================== PATH & KEYS ======================
BASE_DIR = Path(os.path.abspath(__file__)).parents[1]
LOGO_PATH = BASE_DIR / "images" / "LOGO.png"

MOVIES_CSV  = BASE_DIR / "data" / "tmdb_5000_movies.csv"
CREDITS_CSV = BASE_DIR / "data" / "tmdb_5000_credits.csv"

TMDB_API_KEY = "32be515044e4f084aa5b020364d6e780"

BG_PATH = BASE_DIR / "images" / "BG.jpg"
with open(BG_PATH, "rb") as f:
    encoded = b64encode(f.read()).decode()

# ===================== PAGE CONFIG ======================
st.set_page_config(page_title="RCM • Movie Recommender", layout="wide")

# ===================== FONTS + CONTAINER =================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Baskervville:ital,wght@0,400;0,700;1,400&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet">

<style>
:root{ --bg:#FCFAF5; --ink:#1A1A1A; --lime:#D8FF84; --pink:#FFD6E0; --blue:#D6EFFF; }
.block-container{ max-width:1400px; padding-top:50px; }
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded}");
    background-repeat: repeat;
    background-size: cover;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0) !important;
}}
</style>
""", unsafe_allow_html=True)

# ===================== LOGO + NAVIGATION ==========================
col_img, col1, col2 = st.columns([3, 3, 3])

with col_img:    
     logo_path = Path.cwd() / "images" / "LOGO.png"
     if logo_path.exists():
         st.image(str(logo_path), width=250)
     else:
         st.error(f"Không tìm thấy logo: {logo_path}")

with col1:
    if st.button("Homepage", use_container_width=True):
        st.switch_page("homepage.py")            

with col2:
    if st.button("Analyze Your Reviews", use_container_width=True):
        st.switch_page("pages/review.py")

# ====================== STICKER ================================
STICKER_PATH = BASE_DIR / "images" / "sticker 92.png"
with open(STICKER_PATH, "rb") as f:
    sticker_encoded = b64encode(f.read()).decode()

st.markdown("""
<style>
/* ========= HERO CONTAINER (CHA) ========= */
.hero-container {
    width: 100%;
    display: flex;
    justify-content: center;
    position: relative;
}
                
/* ========= STICKER ========= */
.sticker-container {
    width: 280px;
    margin-left: -187px;
}

.sticker-container img {
    width: 100%;
    height: auto;
    display: block;
    # animation: float 3s ease-in-out infinite;
    animation: bob 3s ease-in-out infinite;
}

/* ========= TEXT ========== */
.hero-text {
    position: absolute;
    top: 19%; 
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'Baskervville';
    font-size: 50px;
    font-weight: 500;
    color: var(--ink);
    white-space: nowrap;
}

@media (max-width: 900px) { 
    .sticker-container { width: 280px; }
    .hero-text { font-size: 50px; }
}
@media (max-width: 600px) { 
    .sticker-container { width: 220px; margin-left: -148px; }
    .hero-text { font-size: 40px; top: 19%; }
}

@keyframes bob {
    0%   { transform: translateY(0) rotate(0deg); }
    50%  { transform: translateY(-5px) rotate(-2deg); }
    100% { transform: translateY(0) rotate(0deg); }
}
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-container">
    <div class="sticker-container">
        <img src="data:image/png;base64,{sticker_encoded}">
    </div>
    <div class="hero-text">Your next movie</div>
</div>
""", unsafe_allow_html=True)

# ============== INPUT BOX + RECOMMEND BUTTON + GALLERY ================           
st.markdown("""<style>
/* ======= INPUT BOX ======= */
.stSelectbox label { display:none; }
.stSelectbox > div > div { height:60px; }

/* Style box */
div[data-baseweb="select"] {
    background-color: var(--bg) !important;  /* Keep your preferred blue */
    border: 3px solid var(--ink) !important;
    border-radius: 11px !important;
}
            
/* Text inside */
div[data-baseweb="select"] * {
    background-color: transparent !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 17px !important;
    align-items: center;
}

/* ======= 2 FUNCTION & RECOMMEND BUTTONS ======= */
div[data-testid="stVerticalBlock"] button{
    height:65px; border:3px solid var(--ink); border-radius:11px; background:var(--lime);
    box-shadow:5px 5px 10px 1px var(--pink); transition:transform .15s ease; align-items:center;
}
div[data-testid="stVerticalBlock"] button > * {
    color:var(--ink); font-family:'Courier Prime'; font-weight:700; font-size:20px; line-height:1;
}
div[data-testid="stVerticalBlock"] button:hover{ 
    background:var(--lime); box-shadow:5px 5px 10px 1px var(--pink); transform:scale(1.03); 
}
</style> """, unsafe_allow_html=True)

# ===================== TMDB POSTER ======================
class MyTfidfVectorizer:
    def __init__(self):
        self.vocab_ = {}
        self.idf_ = None

    def _tokenize(self, doc: str):
        # tags của bạn dạng "Action Adventure Tom_Hanks ..." nên split theo space là đủ
        return doc.lower().split()

    def fit_transform(self, docs):
        """
        docs: iterable (list/Series) các chuỗi.
        Trả về: ma trận TF-IDF dạng numpy.ndarray shape (n_docs, n_terms)
        """
        # 1. Tokenize từng doc
        tokenized_docs = [self._tokenize(d) for d in docs]

        # 2. Xây vocab
        vocab = {}
        for tokens in tokenized_docs:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab_ = vocab

        n_docs = len(tokenized_docs)
        n_terms = len(vocab)

        # 3. Tính TF (term frequency) & DF (document frequency)
        X = np.zeros((n_docs, n_terms), dtype=np.float32)
        df_counts = np.zeros(n_terms, dtype=np.int32)

        for i, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)
            for t, c in counts.items():
                j = vocab[t]
                X[i, j] = c
                df_counts[j] += 1

        # 4. Tính IDF: giống công thức phổ biến của sklearn
        # idf_j = log((1 + n_docs) / (1 + df_j)) + 1
        idf = np.log((1.0 + n_docs) / (1.0 + df_counts)) + 1.0
        self.idf_ = idf

        # 5. TF-IDF = TF * IDF
        X *= idf

        # (Có thể chuẩn hóa theo độ dài doc nếu muốn giống sklearn hơn,
        # nhưng ở đây để đơn giản, chuẩn hóa sẽ làm ở bước cosine)
        return X
def cosine_similarity_custom(X: np.ndarray) -> np.ndarray:
    """
    X: ma trận (n_samples, n_features)
    Trả về: ma trận similarity (n_samples, n_samples)
    """
    # Chuẩn hóa từng vector hàng về norm = 1
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Tránh chia cho 0
    norms = np.where(norms == 0, 1e-9, norms)
    X_norm = X / norms

    # Cosine similarity = X_norm * X_norm.T
    return X_norm @ X_norm.T
@st.cache_resource
def load_data():
    import ast

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    df = movies.merge(credits, on="title")

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i["name"])
        return L

    df["genres"] = df["genres"].apply(convert)
    df["keywords"] = df["keywords"].apply(convert)

    def convert_cast(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i["name"])
                counter += 1
        return L

    df["cast"] = df["cast"].apply(convert_cast)

    def get_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return i["name"]
        return ""
        
    df["crew"] = df["crew"].apply(lambda x: [get_director(x)])

    df["tags"] = df["genres"] + df["keywords"] + df["cast"] + df["crew"]
    df["tags"] = df["tags"].apply(lambda x: " ".join(x))
    df = df[["movie_id","title","overview","tags"]]

    # ======= DÙNG TF-IDF & COSINE TỰ CODE =======
    tfidf = MyTfidfVectorizer()
    # .values để đảm bảo là list/array các chuỗi
    vectors = tfidf.fit_transform(df["tags"].values)

    similarity = cosine_similarity_custom(vectors)

    return df.reset_index(drop=True), similarity
movies, similarity = load_data()
all_titles = movies["title"].tolist()

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()
    p = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500{p}" if p else None

def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()

    for v in data.get("results", []):
        if v.get("type") == "Trailer" and v.get("site") == "YouTube":
            key = v.get("key")
            return f"https://www.youtube.com/watch?v={key}"
    return None

# ===================== RECOMMEND GALLERY ===================
def recommend(title, top_k=10):
    # vị trí phim được chọn
    index = movies[movies["title"] == title].index[0]
    # sắp xếp theo độ tương tự giảm dần
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )
    # bỏ phần tử đầu tiên (chính nó), lấy top_k tiếp theo
    picks = []
    for i, _score in distances[1: top_k + 1]:
        row = movies.iloc[i]
        picks.append((row["title"], row["movie_id"]))
    return picks

# ===================== INPUT + BUTTON ===================
c1, cbtn, _ = st.columns([6, 2, 0.1])

with c1:
    selected = st.selectbox("", all_titles, index=None, placeholder="Select a movie")
with cbtn:
    run = st.button("Recommend", use_container_width=True)

if run and not selected:
    st.markdown('<div class="gallery-title">Please select a movie first!</div>', unsafe_allow_html=True)
    run = False

# ===================== GALLERY STYLES ===================
st.markdown(""" 
<style>
/* ======= GALLERY ======= */
.gallery-title{
    text-align:center; font-family:'Courier Prime',monospace; font-weight:700;
    font-size:26px; margin:10px 0 20px; color:var(--ink);
} 
.card{ text-align:center; margin-bottom:30px; position:relative; }

/* 1. CARD IMAGE & LINK STYLES */
.card-link {
    display: block; /* link bao phủ ảnh */
    position: relative; /* Cần thiết để icon (absolute) định vị được */
    text-decoration: none; /* Loại bỏ gạch chân link */
}

.card > .card-img, /* Poster nằm trực tiếp trong .card (No trailer) */
.card-link .card-img { /* Poster nằm trong thẻ link (Có trailer) */
    aspect-ratio: 2 / 3; width:auto; border-radius:16px; box-shadow:0 6px 12px rgba(0,0,0,.25) !important;
    display:block; margin:0 auto; object-fit:cover; transition: transform .25s ease;
}
.card > .card-img:hover, 
.card-link:hover .card-img { transform: scale(1.05); }

.card .caption{
    margin-top:12px; font-family:'Courier Prime',monospace; font-weight:700; font-size:14px;
    letter-spacing:0.5px; color:var(--ink); text-transform:uppercase;
}
    
/* 2. OVERLAY ICONS/TEXT STYLES */
.trailer-btn {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%); 
    min-width: 80px; height: auto;
    padding: 8px 15px;
    border-radius: 10px;
        
    color: white !important;
    font-size: 14px;
    font-family: 'Courier Prime';
    font-weight: 700;
    letter-spacing: 1px;
        
    display: inline-flex; 
    align-items: center !important; 
    justify-content: center;
    white-space: nowrap;
        
    opacity: 0;
    transition: opacity .25s ease, transform .25s ease;
    pointer-events: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    z-index: 10;
            
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(0.5px);
}

.trailer-btn.trailer-active, .no-trailer-active { 
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(0.5px);
}

.card-link:hover .trailer-btn,
.card:hover .trailer-btn {
    opacity: 1; 
    transform: translate(-50%, -50%) scale(1.08); 
}
</style>
""", unsafe_allow_html=True)   

# ===================== RESULTS ==========================
import textwrap
if run:
    recs = recommend(selected, top_k=10)
    if not recs:
        st.warning("No recommendation found for this title.")
    else:
        col_left, col_center, col_right = st.columns([3, 1.5, 3])

        with col_center:
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            center_path = BASE_DIR / "images" / "POINTING.gif"
            if center_path.exists():
                st.image(str(center_path), use_container_width=True)
            else:
                st.error(f"Không tìm thấy POINTING.gif tại: {center_path}")
        
        with col_left:
            st.markdown('<div style="height: 49px;"></div>', unsafe_allow_html=True)
            left_path = BASE_DIR / "images" / "REC.png"
            if left_path.exists():
                st.image(str(left_path), width=550)
        
        with col_right:
            st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
            right_path = BASE_DIR / "images" / "MOV.png"
            if right_path.exists():
                st.image(str(right_path), width=550)

        rows = [recs[:5], recs[5:10]]

        for row in rows:
            cols = st.columns(5, gap="large")

            for col, (title, mid) in zip(cols, row):
                with col:
                    poster = fetch_poster(int(mid))
                    trailer = fetch_trailer(int(mid))

                    if poster:
                        if trailer:
                            raw_html = f"""
                                <div class="card">
                                    <a href="{trailer}" target="_blank" class="card-link">
                                        <img src="{poster}" alt="{title}" class="card-img">
                                        <span class="trailer-btn trailer-active">▶ Trailer</span>
                                    </a>
                                    <div class="caption">{title}</div>
                                </div>
                            """
                        else: # ko có trailer
                            raw_html = f"""
                                <div class="card">
                                    <div class="card-link no-link-wrapper">
                                        <img src="{poster}" alt="{title}" class="card-img">
                                        <span class="trailer-btn no-trailer-active">⊘ No Trailer</span>
                                    </div>
                                    <div class="caption">{title}</div>
                                </div>
                            """
                        card_html = textwrap.dedent(raw_html)
                        st.markdown(card_html, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: center; margin-top: 30px; ">
                <span style="font-family: 'Courier Prime'; font-weight: 700; font-size: 35px; color: var(--ink); ">
                ✨ HAVE A GREAT MOVIE NIGHT ✨
                </span>
            </div>""", unsafe_allow_html=True)
