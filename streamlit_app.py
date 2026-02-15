"""
Movie Recommendation System - Streamlit Frontend
================================================
Connects to the FastAPI backend and renders a clean UI
for getting personalized movie recommendations.
"""

import streamlit as st
import requests
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0A1628; color: #E2E8F0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0D2B55; }

    /* Cards */
    .movie-card {
        background: #112240;
        border: 1px solid #1A3A60;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateY(-2px); border-color: #0EA5C8; }

    .rank-badge {
        display: inline-block;
        background: #0B6E8C;
        color: white;
        border-radius: 50%;
        width: 32px; height: 32px;
        text-align: center; line-height: 32px;
        font-weight: bold; font-size: 14px;
        margin-right: 10px;
    }

    .score-bar-container {
        background: #1A3A60;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }
    .score-bar {
        background: linear-gradient(90deg, #0B6E8C, #10C2A0);
        border-radius: 6px;
        height: 8px;
    }

    .genre-tag {
        display: inline-block;
        background: #0D2B55;
        border: 1px solid #0B6E8C;
        color: #0EA5C8;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px;
    }

    .stat-card {
        background: #112240;
        border: 1px solid #1A3A60;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stat-number { font-size: 28px; font-weight: bold; color: #0EA5C8; }
    .stat-label  { font-size: 12px; color: #A8C4D4; margin-top: 4px; }

    /* Headers */
    h1, h2, h3 { color: #FFFFFF !important; }
    .stMarkdown p { color: #CBD5E1; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0B6E8C, #10C2A0);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-size: 15px; font-weight: bold;
        width: 100%; cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Sliders and inputs */
    .stSlider > div > div { background: #0B6E8C; }
    label { color: #A8C4D4 !important; }
</style>
""", unsafe_allow_html=True)


# ── API Helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def fetch_recommendations(user_id: int, top_k: int, top_n: int):
    try:
        payload = {"user_id": user_id, "top_k_retrieve": top_k, "top_n_final": top_n}
        r = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
        if r.ok:
            return r.json(), None
        return None, r.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure FastAPI is running on port 8000."
    except Exception as e:
        return None, str(e)


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.ok
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("---")

    # API Status
    api_ok = check_api_health()
    if api_ok:
        st.success("✅  API Connected")
    else:
        st.error("API Offline")
        st.info("Run: `uvicorn api.main:app --reload`")

    st.markdown("---")
    st.markdown("### Settings")

    user_id = st.number_input(
        "User ID",
        min_value=1, max_value=6040, value=1, step=1,
        help="Enter a user ID between 1 and 6040"
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=5, max_value=30, value=10, step=5,
    )

    top_k = st.slider(
        "Retrieval Candidates (Stage 1)",
        min_value=50, max_value=500, value=200, step=50,
        help="More candidates = better quality but slower"
    )

    st.markdown("---")

    recommend_btn = st.button("🚀 Get Recommendations", use_container_width=True)

    st.markdown("---")

    # Stats in sidebar
    stats = fetch_stats()
    if stats:
        st.markdown("### System Info")
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:8px">
            <div class="stat-number">{stats['total_users']:,}</div>
            <div class="stat-label">Total Users</div>
        </div>
        <div class="stat-card" style="margin-bottom:8px">
            <div class="stat-number">{stats['total_movies']:,}</div>
            <div class="stat-label">Total Movies</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats['total_genres']}</div>
            <div class="stat-label">Genres</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════
st.markdown("# 🎬 Movie Recommendation System")
st.markdown("##### Two-Stage Retrieval + Ranking · MovieLens 1M · TensorFlow / Keras")
st.markdown("---")

# Show pipeline explanation
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:24px">🔍</div>
        <div style="font-weight:bold; color:#0EA5C8; margin:4px 0">Stage 1: Retrieval</div>
        <div style="font-size:12px; color:#A8C4D4">
        Two-Tower network computes cosine similarity → Top-K candidates
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:24px">🏆</div>
        <div style="font-weight:bold; color:#10C2A0; margin:4px 0">Stage 2: Ranking</div>
        <div style="font-size:12px; color:#A8C4D4">
        Deep ranking model predicts exact rating → Sorts by score
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:24px">🎯</div>
        <div style="font-weight:bold; color:#F59E0B; margin:4px 0">Final Output</div>
        <div style="font-size:12px; color:#A8C4D4">
        Personalized Top-N movies with predicted ratings
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Results ─────────────────────────────────────────────────────────────
if recommend_btn:
    if not api_ok:
        st.error("Cannot connect to the API. Please start the FastAPI server first.")
    else:
        with st.spinner(f"Fetching recommendations for User {user_id}..."):
            data, error = fetch_recommendations(user_id, top_k, top_n)

        if error:
            st.error(f"Error: {error}")

        elif data:
            recs = data["recommendations"]

            # User profile header
            st.markdown(f"""
            ### 👤 User {data['user_id']} Profile
            """)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Gender",     data["gender"])
            p2.metric("Age",        data["age"])
            p3.metric("Occupation", f"#{data['occupation']}")
            p4.metric("Results",    f"{len(recs)} movies")

            st.markdown(f"---")
            st.markdown(f"### 🎬 Top {len(recs)} Recommendations")

            # Results in 2 columns
            col_left, col_right = st.columns(2)
            for i, movie in enumerate(recs):
                col = col_left if i % 2 == 0 else col_right

                # Score as 0-5 scale for display
                score_norm = max(0, min(1, movie["ranking_score"] / 5.0))
                ret_norm   = max(0, min(1, movie["retrieval_score"]))

                genres_html = "".join(
                    f'<span class="genre-tag">{g}</span>'
                    for g in movie["genres"].split("|")
                )

                year_str = str(movie["year"]) if movie["year"] else "N/A"

                with col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display:flex; align-items:center; margin-bottom:8px">
                            <span class="rank-badge">#{movie['rank']}</span>
                            <span style="font-size:15px; font-weight:bold; color:#FFFFFF">
                                {movie['title']}
                            </span>
                        </div>
                        <div style="margin-bottom:8px">{genres_html}</div>
                        <div style="font-size:12px; color:#A8C4D4; margin-bottom:10px">
                            {year_str}
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:3px">
                            <span style="font-size:12px; color:#A8C4D4">Predicted Rating</span>
                            <span style="font-size:12px; font-weight:bold; color:#10C2A0">
                                 {movie['ranking_score']:.2f} / 5
                            </span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width:{score_norm*100:.1f}%"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px; margin-bottom:3px">
                            <span style="font-size:11px; color:#64748B">Retrieval Score</span>
                            <span style="font-size:11px; color:#64748B">
                                {movie['retrieval_score']:.4f}
                            </span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar"
                                 style="width:{ret_norm*100:.1f}%; background: linear-gradient(90deg,#0B6E8C,#0EA5C8)">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Download CSV
            st.markdown("---")
            df_export = pd.DataFrame([{
                "Rank":            m["rank"],
                "Movie ID":        m["movie_id"],
                "Title":           m["title"],
                "Genres":          m["genres"],
                "Year":            m["year"],
                "Predicted Rating": m["ranking_score"],
                "Retrieval Score": m["retrieval_score"],
            } for m in recs])

            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Recommendations as CSV",
                data=csv,
                file_name=f"recommendations_user_{user_id}.csv",
                mime="text/csv",
            )

else:
    # Placeholder when no recommendation yet
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:#112240;
                border-radius:12px; border:1px dashed #1A3A60;">
        <div style="font-size:48px; margin-bottom:16px">🎬</div>
        <div style="font-size:20px; font-weight:bold; color:#FFFFFF; margin-bottom:8px">
            Ready to recommend!
        </div>
        <div style="color:#A8C4D4; font-size:14px">
            Select a User ID from the sidebar and click
            <strong style="color:#0EA5C8">Get Recommendations</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)