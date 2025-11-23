import streamlit as st
import pandas as pd
import requests
import base64
from pathlib import Path

#Title
st.set_page_config(page_title="Movie Analytic & Recommendation", page_icon="🎬", layout="wide")


#Background
image_path = Path.cwd() / "images" / "BG.jpg"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
image_url = f"data:image/jpeg;base64,{encoded_image}"

#Add styles(CSS)
st.markdown("""
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Baskervville:ital,wght@0,400;0,700;1,400&family=Courier+Prime:wght@400;700&family=Caveat:wght@700&display=swap" rel="stylesheet">

    <style>
    /* Root & Layout */
    :root{ --ink:#1A1A1A; --lime:#D8FF84; --pink:#FFD6E0; --blue:#D6EFFF; }

    [data-testid="stAppViewContainer"]{
        background: transparent !important;
    }

    .block-container{
        max-width:1400px;
        padding-top:80px;
    }
    
    /* Subtitle Box */
    .subtitle-box{
        font-family:'Baskervville', cursive;
        font-size:40px;
        text-align:center;
        color:#1A1A1A;
        margin-top:20px;
    }

    .highlight{
        background-color:#FFD6E0;
        padding:4px 10px;
        border-radius:6px;
    }

    /* Typography */
    body, .stApp{
        background-color:var(--bg);
        color:var(--ink);
        font-family:'Georgia', serif;
    }

    h1, h2, h3{
        font-family:'Baskervville', cursive, sans-serif;
        font-weight:bold;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"]{
        background-color:var(--blue) !important;
        border-radius:12px !important;
        border:2px solid var(--ink) !important;
        padding:12px !important;
        font-family:'Courier Prime', monospace;
    }
    div[data-testid="stFileUploader"] *{
        background-color:transparent !important;
    }

    /* Textarea */
    div[data-testid="stTextArea"] textarea {
        font-family:'Baskervville', cursive !important;
        font-size:20px !important;
        color:var(--ink) !important;
        background-color:#FFFBEA !important;
        border: 2px solid #000 !important;
        border-radius:10px !important;
        padding:10px !important;
    }

    /* Result Box */
    .result-box{
        background-color:#FFFFFF;
        border:2px solid var(--ink);
        border-radius:10px;
        padding:25px 10px;
        text-align:center;
        font-family:'Courier Prime', monospace;
        font-weight:800;
        font-size:18px;
        line-height:1.8;
        box-shadow:4px 4px 0px var(--lime);
        width:250px;
        margin:0 auto;
        position:relative;
        top:40px;
    }
    
    /* Dataframe */
    div[data-testid="stDataFrame"] table {
        border: 2px solid #000 !important;
        border-collapse: collapse !important;
    }

    div[data-testid="stDataFrame"] table th, 
    div[data-testid="stDataFrame"] table td {
        border: 1px solid #000 !important;
        padding: 8px 12px !important;
    }

    /* Buttons */
    div[data-testid="stVerticalBlock"] button{
        height:65px;
        border:3px solid var(--ink);
        border-radius:11px;
        background:var(--lime);
        box-shadow:5px 5px 10px 1px var(--pink);
        transition:transform .15s ease;
    }
    div[data-testid="stVerticalBlock"] button > *{
        color:var(--ink);
        font-family:'Courier Prime', monospace;
        font-weight:700;
        font-size:20px;
    }
    div[data-testid="stVerticalBlock"] button:hover{
        transform:scale(1.03);
        box-shadow:5px 5px 10px 1px var(--pink);
    }
    div[data-testid="stVerticalBlock"] button {
        background-color: var(--lime) !important;
        opacity: 1 !important;
        backdrop-filter: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("{image_url}") no-repeat center center fixed !important;
        background-size: cover !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
    }}
    </style>
""", unsafe_allow_html=True)

#Alerts
def show_alert(message: str, kind: str = "info"):
    #Colors
    colors = {
        "info": "#e8f4fd",
        "success": "#e6f4ea",
        "warning": "#fff4e5",
        "error": "#fdecea"
    }
    color = colors.get(kind.lower(), "#e8f4fd")

    # HTML alert
    alert_html = f"""
    <div class="alert" style="
        position: relative;
        padding: 16px;
        margin: 12px 0;
        border: 2px solid #000 !important;
        border-radius: 8px;
        background-color: {color};
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        z-index: 9999;
        isolation: isolate;
        font-weight: 500;
        font-size: 18px;
    ">
        {message}
    </div>
    """
    st.markdown(alert_html, unsafe_allow_html=True)

#Navigation
col_img, col1, col2 = st.columns([2, 2, 2])
with col_img:
    logo_path = Path.cwd() / "images" / "LOGO.png"
    st.image(str(logo_path), width=300)
with col1:
    if st.button("Homepage", use_container_width=True):
        st.switch_page("homepage.py")            

with col2:
    if st.button("Movies Recommendations", use_container_width=True):
        st.switch_page("pages/recommendations.py") 

st.markdown("""
<div class="subtitle-box">
    Help you <span class="highlight">Analyze</span> your movies
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

#API
API_URL = "https://group-8-dseb66b-analyze-reviews-and.onrender.com/predict"

#Input raw text & upload file
left, center, right = st.columns([1, 6, 1])
with center:
    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown("<div style='font-family:Courier prime, cursive; font-size:30px;'>Write your movie review here</div>", unsafe_allow_html=True)
        raw_text = st.text_area("", placeholder="Text area...", height=250)
    with col4:
        st.markdown("<div style='font-family:Courier prime, cursive; font-size:30px;'>Your review file</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["csv"])

#Data process, Data preview display
df = pd.DataFrame()

left, center, right = st.columns([1, 6, 1])
with center:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df_preview = df.copy()
        df_preview = df_preview.loc[:, ~df_preview.columns.str.contains('^No$|^Unnamed')]
        st.write("#### Data preview")
        st.dataframe(df_preview.head(), hide_index=True)

    elif raw_text:
        lines = raw_text.strip().split('\n')
        df = pd.DataFrame({"Review": lines})
        st.write("#### Your review:")
        st.dataframe(df, hide_index=True)
    Analyze = st.button(label = "Analyze")

#Send Data to API
alert_placeholder = st.empty()
if Analyze:
    #If user import text
    if raw_text and uploaded_file is None:
        lines = raw_text.strip().split('\n')
        lines = [line for line in lines if line.strip()]
        input_data = {"text": lines}
        
        with alert_placeholder:
            left, center, right = st.columns([1, 6, 1])
            with center:
                show_alert(f"⏳ Sending data to API... please wait a moment.", kind="info")
        try:
            result = requests.post(API_URL, json=input_data)
            response_json = result.json()
            alert_placeholder.empty()
            
            if isinstance(response_json, list) and len(response_json) > 0:
                #Convert list dict to DataFrame
                first_result = pd.DataFrame(response_json)
                first_result.reset_index(inplace=True)
                first_result.rename(columns={"index": "No"}, inplace=True)
                first_result["No"] = first_result["No"] + 1
                #Results Display
                if "pred" in first_result.columns:
                    counts = first_result["pred"].str.lower().value_counts(dropna=False)
                    pos = int(counts.get("positive", 0))
                    neg = int(counts.get("negative", 0))
            
                    total = len(first_result)
                    pos_rate = pos / total if total else 0.0
                    neg_rate = neg / total if total else 0.0
                left, center, right = st.columns([1, 6, 1])
                with center:
                    show_alert(f"✅ Analyzis successful!", kind="success")
                    st.markdown("### Results Table")
                    col5, col6 = st.columns([2, 1])
                    with col5:
                        st.dataframe(first_result[["No", "review", "pred", "score"]], hide_index=True)
                    with col6:
                        like = Path.cwd() / "images" / "LIKE.gif"
                        disklike = Path.cwd() / "images" / "DISLIKE.gif"
                        if pos_rate >= 0.5:
                            st.markdown(
                                f"""
                                <div style="text-align:center;">
                                    <img src="data:image/gif;base64,{base64.b64encode(open(like, "rb").read()).decode()}" width="200">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="text-align:center;">
                                    <img src="data:image/gif;base64,{base64.b64encode(open(disklike, "rb").read()).decode()}" width="200">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown(
                            f"""
                            <div class="result-box">
                                POSITIVE: {pos} ({pos_rate:.1%})<br>
                                NEGATIVE: {neg} ({neg_rate:.1%})
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                left, center, right = st.columns([1, 6, 1])
                with center:
                    show_alert(f"⚠️ API returned invalid format.", kind="warning")
        except Exception as e:
            left, center, right = st.columns([1, 6, 1])
            with center:
                show_alert(f"🚫 API connection error: {e}", kind="error")

    #If user import file
    elif uploaded_file is not None:
        try:
            #Assume the first column contains reviews
            text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            texts = df[text_col].astype(str).tolist()

            input_data = {"text": texts}
            with alert_placeholder:
                left, center, right = st.columns([1, 6, 1])
                with center:
                    show_alert(f"⏳ Sending data to API... please wait a moment.", kind="info")
            result = requests.post(API_URL, json=input_data, timeout=60)
            response_json = result.json()
            alert_placeholder.empty()

            if isinstance(response_json, list):
                #Convert list dict to DataFrame
                df_result = pd.DataFrame(response_json)
                df_result.reset_index(inplace=True)
                df_result.rename(columns={"index": "No"}, inplace=True)
                df_result["No"] = df_result["No"] + 1

                #Results Display
                left, center, right = st.columns([1, 6, 1])
                with center:
                    show_alert(f"✅ Analyzis successful!", kind="success")
                if "pred" in df_result.columns:
                    counts = df_result["pred"].str.lower().value_counts(dropna=False)
                    pos = int(counts.get("positive", 0))
                    neg = int(counts.get("negative", 0))
            
                    total = len(df_result)
                    pos_rate = pos / total if total else 0.0
                    neg_rate = neg / total if total else 0.0
        

                left, center, right = st.columns([1, 6, 1])
                with center:
                    st.markdown("### Results Table")
                left, center, right = st.columns([1, 6, 1])
                with center:
                    col7, col8 = st.columns([2, 1])
                    with col7:
                        st.dataframe(df_result[["No", "review", "pred", "score"]], hide_index=True)

                    with col8:
                        like = Path.cwd() / "images" / "LIKE.gif"
                        disklike = Path.cwd() / "images" / "DISLIKE.gif"
                        if pos_rate >= 0.5:
                            st.markdown(
                                f"""
                                <div style="text-align:center;">
                                    <img src="data:image/gif;base64,{base64.b64encode(open(like, "rb").read()).decode()}" width="200">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="text-align:center;">
                                    <img src="data:image/gif;base64,{base64.b64encode(open(disklike, "rb").read()).decode()}" width="200">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown(
                            f"""
                            <div class="result-box">
                                POSITIVE: {pos} ({pos_rate:.1%})<br>
                                NEGATIVE: {neg} ({neg_rate:.1%})
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                left, center, right = st.columns([1, 6, 1])
                with center:
                    show_alert(f"⚠️ API returned invalid format.", kind="warning")
        except Exception as e:
            left, center, right = st.columns([1, 6, 1])
            with center:
                show_alert(f"🚫 API connection error: {e}", kind="error")
