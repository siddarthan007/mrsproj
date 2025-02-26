import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

@st.cache_data
def load_data():
    try:
        save_dict = joblib.load("models/recommender_model.joblib")  
        df = save_dict['df']  
        feature_matrix = save_dict['feature_matrix']  
        knn_model = save_dict['knn_model']  
        movie_ids = save_dict['movie_ids']  
        return df, feature_matrix, knn_model, movie_ids
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

df, feature_matrix, knn_model, movie_ids = load_data()
if df is None or feature_matrix is None or knn_model is None:
    st.stop()

def fetch_poster(poster_path):
    if pd.isna(poster_path) or not poster_path:
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

def get_recommendations(selected_indices, nn_model, matrix, n_recommend=10):
    try:
        distances, indices = nn_model.kneighbors(matrix[selected_indices])
        score_dict = {}
        for dist, idx in zip(distances, indices):
            for d, i in zip(dist, idx):
                if i not in selected_indices:  
                    similarity = 1 - d
                    score_dict[i] = score_dict.get(i, 0) + similarity
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_scores[:n_recommend]]
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #FF6347; }
        .sub-title { text-align: center; font-size: 20px; color: #4682B4; }
        .movie-container { display: flex; flex-direction: column; align-items: center; text-align: center; }
        .movie-title { font-size: 16px; font-weight: bold; margin-top: 10px; width: 150px; }
        .tmdb-button { margin-top: 5px; font-size: 12px; color: #4682B4; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Recommendation System 🍿</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Select up to 5 movies to get personalized recommendations!</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

debug_mode = st.toggle("Enable Debug Mode", value=False)

if 'selected_movies' not in st.session_state:
    st.session_state['selected_movies'] = []

search_query = st.text_input("Search for a movie:", placeholder="Type a movie name...")
if search_query:
    filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)].head(20)
else:
    filtered_df = pd.DataFrame()

if not filtered_df.empty:
    st.subheader("🔍 Search Results")
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        movie_id = row['movieId']
        poster_url = fetch_poster(row['poster_path'])
        with cols[i % 4]:
            with st.container():
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(poster_url, width=150)
                st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if movie_id in st.session_state['selected_movies']:
                    st.button("✅ Selected", disabled=True, key=f"sel_{i}")
                elif len(st.session_state['selected_movies']) < 5:
                    if st.button("➕ Select", key=f"sel_btn_{i}"):
                        st.session_state['selected_movies'].append(movie_id)
                        st.rerun()

st.subheader(f"🎥 Selected Movies ({len(st.session_state['selected_movies'])}/5)")
st.markdown("<hr>", unsafe_allow_html=True)
if st.session_state['selected_movies']:
    cols = st.columns(5)
    for i, movie_id in enumerate(st.session_state['selected_movies']):
        movie = df[df['movieId'] == movie_id].iloc[0]
        poster_url = fetch_poster(movie['poster_path'])
        with cols[i % 5]:
            with st.container():
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(poster_url, width=150)
                st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if st.button("❌ Remove", key=f"rem_{i}"):
                    st.session_state['selected_movies'].remove(movie_id)
                    st.rerun()
else:
    st.info("No movies selected yet.")

if len(st.session_state['selected_movies']) >= 3:
    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            selected_indices = [np.where(movie_ids == mid)[0][0] for mid in st.session_state['selected_movies']]
            rec_indices = get_recommendations(selected_indices, knn_model, feature_matrix)
            if rec_indices:
                st.subheader("🎬 Recommended Movies for You:")
                st.markdown("<hr>", unsafe_allow_html=True)
                cols = st.columns(5)
                for i, idx in enumerate(rec_indices[:10]):
                    movie = df.iloc[idx]
                    poster_url = fetch_poster(movie['poster_path'])
                    with cols[i % 5]:
                        with st.container():
                            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                            st.image(poster_url, width=150)
                            st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please select at least 3 movies to get recommendations.")

if debug_mode and st.session_state['selected_movies']:
    with st.sidebar:
        st.subheader("Debug Information")
        selected_indices = [np.where(movie_ids == mid)[0][0] for mid in st.session_state['selected_movies']]
        rec_indices = get_recommendations(selected_indices, knn_model, feature_matrix, n_recommend=10)

        if len(rec_indices) > 1:
            rec_features = feature_matrix[rec_indices]
            sim_matrix = cosine_similarity(rec_features)
            diversity = 1 - sim_matrix.mean()
            st.write(f"**Diversity**: {diversity:.2f} (average dissimilarity between recommendations)")
        else:
            st.write("**Diversity**: N/A (insufficient recommendations)")

        if st.checkbox("Visualize Feature Space", value=False):
            with st.spinner("Generating 2D projection..."):
                    combined_indices = selected_indices + rec_indices
                    relevant_features = feature_matrix[combined_indices]
                    reducer = umap.UMAP(random_state=42)
                    embedding = reducer.fit_transform(relevant_features)
                    num_selected = len(selected_indices)
                    selected_embedding = embedding[:num_selected]
                    recommended_embedding = embedding[num_selected:]
                    fig, ax = plt.subplots()
                    ax.scatter(selected_embedding[:, 0], selected_embedding[:, 1], color='red', label='Selected', s=50)
                    ax.scatter(recommended_embedding[:, 0], recommended_embedding[:, 1], color='green', label='Recommended', s=50)
                    ax.legend()
                    st.pyplot(fig)

        if st.checkbox("Show Similarity Heatmap", value=False):
            sim_matrix = cosine_similarity(feature_matrix[selected_indices], feature_matrix[rec_indices])
            fig, ax = plt.subplots()
            sns.heatmap(sim_matrix, annot=True, cmap="YlGnBu", ax=ax)
            ax.set_xticklabels([df.iloc[idx]['title'][:15] + '...' for idx in rec_indices], rotation=45, ha='right')
            ax.set_yticklabels([df.iloc[idx]['title'][:15] + '...' for idx in selected_indices], rotation=0)
            st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888;">Built using TMDB dataset and KNN with Cosine similarity.</div>', unsafe_allow_html=True)