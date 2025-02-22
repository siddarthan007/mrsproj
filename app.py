import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

@st.cache_resource
def load_models():
    return joblib.load("models/mrs_model_v11.pkl.gz")

df, feature_matrix, nn_cosine = load_models()

def fetch_poster(poster_path):
    if not poster_path or pd.isna(poster_path):
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

def get_recommendations(selected_indices, nn_model, matrix, n_recommend=10):
    matrix = matrix.tocsr()
    distances, indices = nn_model.kneighbors(matrix[selected_indices, :])
    score_dict = {}
    for dist, idx in zip(distances, indices):
        for d, i in zip(dist, idx):
            if i not in selected_indices:
                similarity = 1 - d
                score_dict[i] = score_dict.get(i, 0) + similarity
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_scores[:n_recommend]]

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #FF4B4B;
        }
        .sub-title {
            text-align: center;
            font-size: 20px;
            color: #777;
        }
        .movie-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .movie-title {
            font-size: 16px;
            font-weight: bold;
            margin-top: 10px;
            width: 150px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Recommendation System 🍿</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Search and select up to 5 movies you like!</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []

def search_movies(query):
    if not query:
        return df.iloc[0:0]
    return df[df["original_title"].str.contains(query, case=False, na=False) & df["poster_path"].notna()].head(20)

search_query = st.text_input("Search for a movie:", placeholder="Type a movie name...")
search_results = search_movies(search_query)

if not search_results.empty:
    st.subheader("🔍 Search Results")
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (_, row) in enumerate(search_results.iterrows()):
        movie_title = row["original_title"]
        poster_url = fetch_poster(row["poster_path"])
        with cols[i % 4]:
            with st.container():
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(poster_url, width=150)
                st.markdown(f'<div class="movie-title">{movie_title}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if movie_title in st.session_state.selected_movies:
                    st.button("✅ Selected", disabled=True, key=f"sel_{i}")
                elif len(st.session_state.selected_movies) < 5:
                    if st.button("➕ Select", key=f"sel_btn_{i}"):
                        st.session_state.selected_movies.append(movie_title)

st.subheader("🎥 Selected Movies")
st.markdown("<hr>", unsafe_allow_html=True)
cols = st.columns(5)
for i, movie in enumerate(st.session_state.selected_movies):
    movie_row = df[df["original_title"] == movie].iloc[0]
    poster_url = fetch_poster(movie_row["poster_path"])
    with cols[i % 5]:
        with st.container():
            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
            st.image(poster_url, width=150)
            st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("❌ Remove", key=f"rem_{i}"):
                st.session_state.selected_movies.remove(movie)
                st.rerun()

if len(st.session_state.selected_movies) >= 3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎯 Get Recommendations", use_container_width=True):
        st.markdown("<hr>", unsafe_allow_html=True)
        selected_indices = [df[df["original_title"] == movie].index[0] for movie in st.session_state.selected_movies]
        rec_indices = get_recommendations(selected_indices, nn_cosine, feature_matrix)
        st.subheader("🎬 Recommended Movies for You:")
        st.markdown("<hr>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, movie_idx in enumerate(rec_indices[:10]):
            movie_title = df.iloc[movie_idx]["original_title"]
            poster_url = fetch_poster(df.iloc[movie_idx]["poster_path"])
            with cols[i % 5]:
                with st.container():
                    st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                    st.image(poster_url, width=150)
                    st.markdown(f'<div class="movie-title">{movie_title}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888;">Built using TMDB dataset and KNN models with Cosine similarity.</div>', unsafe_allow_html=True)