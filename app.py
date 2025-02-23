import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

@st.cache_resource
def load_models():
    try:
        return joblib.load("models/mrs_model_v11.pkl.gz")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

df, feature_matrix, nn_cosine = load_models()
if df is None or feature_matrix is None or nn_cosine is None:
    st.stop()

def fetch_poster(poster_path):
    if not poster_path or pd.isna(poster_path):
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

def get_recommendations(selected_indices, nn_model, matrix, n_recommend=10, similarity_threshold=0.3):
    try:
        matrix = matrix.tocsr()
        distances, indices = nn_model.kneighbors(matrix[selected_indices, :])
        score_dict = {}
        for dist, idx in zip(distances, indices):
            for d, i in zip(dist, idx):
                if i not in selected_indices:
                    similarity = 1 - d
                    if similarity > similarity_threshold:
                        score_dict[i] = score_dict.get(i, 0) + similarity
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_scores[:n_recommend]]
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

def create_nn_model(matrix, n_neighbors):
    return NearestNeighbors(metric="cosine", n_neighbors=n_neighbors).fit(matrix)

def evaluate_selected_k(selected_indices, nn_model, matrix, n_recommend=10, similarity_threshold=0.3):
    try:
        rec_indices = get_recommendations(selected_indices, nn_model, matrix, n_recommend, similarity_threshold)
        if not rec_indices:
            return {"precision": 0, "diversity": 0, "coverage": 0}

        matrix = matrix.tocsr()

        selected_features = matrix[selected_indices, :]
        rec_features = matrix[rec_indices, :]
        similarities = []
        for sel_idx in selected_indices:
            sel_vector = matrix[sel_idx, :].toarray().flatten()
            for rec_idx in rec_indices:
                rec_vector = matrix[rec_idx, :].toarray().flatten()
                cosine_sim = np.dot(sel_vector, rec_vector) / (np.linalg.norm(sel_vector) * np.linalg.norm(rec_vector))
                if not np.isnan(cosine_sim):
                    similarities.append(cosine_sim)
        precision = np.mean(similarities) if similarities else 0

        if len(rec_indices) > 1:
            pairwise_dists = []
            for i in range(len(rec_indices)):
                for j in range(i + 1, len(rec_indices)):
                    vec_i = rec_features[i].toarray().flatten()
                    vec_j = rec_features[j].toarray().flatten()
                    dist = 1 - (np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)))
                    if not np.isnan(dist):
                        pairwise_dists.append(dist)
            diversity = np.mean(pairwise_dists) if pairwise_dists else 0
        else:
            diversity = 0

        all_genres = set(df["genres"].explode().unique())
        rec_genres = set(df.iloc[rec_indices]["genres"].explode().unique())
        coverage = len(rec_genres) / len(all_genres) if all_genres else 0

        return {"precision": precision, "diversity": diversity, "coverage": coverage}

    except Exception as e:
        st.error(f"Error evaluating k: {e}")
        return {"precision": 0, "diversity": 0, "coverage": 0}

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
        .tmdb-button {
            margin-top: 5px;
            font-size: 12px;
            color: #007bff;
            text-decoration: underline;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Recommendation System 🍿</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Search and select up to 5 movies you like to get personalized recommendations!</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []
if "nn_model" not in st.session_state:
    st.session_state.nn_model = nn_cosine

debug_mode = st.checkbox("Enable Debug Mode", value=False)

if debug_mode:
    k_neighbors = st.slider("Adjust number of neighbors (k) for recommendations:", min_value=5, max_value=100, value=20, step=5)

    if st.button("Update Model with New k", use_container_width=True):
        with st.spinner("Updating model..."):
            st.session_state.nn_model = create_nn_model(feature_matrix, k_neighbors)
        st.success(f"Model updated with k = {k_neighbors}")

    if st.button("Evaluate Current k", use_container_width=True) and st.session_state.selected_movies:
        with st.spinner("Evaluating current k value..."):
            results = evaluate_selected_k(st.session_state.selected_movies, st.session_state.nn_model, feature_matrix)
            st.subheader(f"Evaluation Results for k = {k_neighbors}")
            st.write(f"Precision = **{results['precision']:.2f}**, Diversity = **{results['diversity']:.2f}**, Coverage = **{results['coverage']:.2f}**")
            st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.checkbox("Visualize Feature Space", value=False) and st.session_state.selected_movies:
        with st.spinner("Generating 2D projection..."):
            reducer = umap.UMAP(random_state=42)
            sample_indices = st.session_state.selected_movies + get_recommendations(st.session_state.selected_movies, st.session_state.nn_model, feature_matrix, n_recommend=10)
            sample_matrix = feature_matrix[sample_indices].toarray()
            embedding = reducer.fit_transform(sample_matrix)
            
            fig, ax = plt.subplots()
            ax.scatter(embedding[:len(st.session_state.selected_movies), 0], embedding[:len(st.session_state.selected_movies), 1], c='blue', label='Selected', s=100)
            ax.scatter(embedding[len(st.session_state.selected_movies):, 0], embedding[len(st.session_state.selected_movies):, 1], c='red', label='Recommended', s=50)
            ax.legend()
            st.pyplot(fig)

    if st.checkbox("Show Similarity Heatmap", value=False) and st.session_state.selected_movies:
        rec_indices = get_recommendations(st.session_state.selected_movies, st.session_state.nn_model, feature_matrix)
        sim_matrix = np.zeros((len(st.session_state.selected_movies), len(rec_indices)))
        for i, sel_idx in enumerate(st.session_state.selected_movies):
            sel_vector = feature_matrix[sel_idx].toarray().flatten()
            for j, rec_idx in enumerate(rec_indices):
                rec_vector = feature_matrix[rec_idx].toarray().flatten()
                sim_matrix[i, j] = np.dot(sel_vector, rec_vector) / (np.linalg.norm(sel_vector) * np.linalg.norm(rec_vector))
        
        fig, ax = plt.subplots()
        sns.heatmap(sim_matrix, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Recommended Movies")
        ax.set_ylabel("Selected Movies")
        st.pyplot(fig)

    if st.checkbox("Show Feature Importance", value=False) and st.session_state.selected_movies:
        sel_idx = st.session_state.selected_movies[0]
        feature_vector = feature_matrix[sel_idx].toarray().flatten()
        feature_names = df.columns 
        top_features = sorted(zip(feature_names, feature_vector), key=lambda x: x[1], reverse=True)[:5]
        st.write("Top 5 Contributing Features for First Selected Movie:")
        for name, value in top_features:
            st.write(f"{name}: {value:.3f}")

    if st.checkbox("Check Feature Matrix Sparsity", value=False):
        sparsity = 1 - feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])
        st.write(f"Feature Matrix Sparsity: {sparsity:.2%}")

    if st.checkbox("Show Neighbor Distance Distribution", value=False) and st.session_state.selected_movies:
        distances, _ = st.session_state.nn_model.kneighbors(feature_matrix[st.session_state.selected_movies])
        fig, ax = plt.subplots()
        ax.hist(distances.flatten(), bins=20)
        ax.set_xlabel("Cosine Distance")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    if st.checkbox("Preview Recommendations", value=False) and st.session_state.selected_movies:
        with st.spinner("Generating preview..."):
            rec_indices = get_recommendations(st.session_state.selected_movies, st.session_state.nn_model, feature_matrix)
            st.write("Preview Recommendations:")
            for idx in rec_indices[:5]:
                st.write(df.iloc[idx]["original_title"])

def search_movies(query):
    if not query.strip():
        return df.iloc[0:0]
    return df[df["original_title"].str.contains(query, case=False, na=False) & 
              df["poster_path"].notna()].head(20)

search_query = st.text_input("Search for a movie:", placeholder="Type a movie name...")
with st.spinner("Searching..."):
    search_results = search_movies(search_query)

if not search_results.empty:
    st.subheader("🔍 Search Results")
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (_, row) in enumerate(search_results.iterrows()):
        movie_idx = row.name
        movie_title = row["original_title"]
        poster_url = fetch_poster(row["poster_path"])
        with cols[i % 4]:
            with st.container():
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(poster_url, width=150)
                st.markdown(f'<div class="movie-title">{movie_title}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if movie_idx in st.session_state.selected_movies:
                    st.button("✅ Selected", disabled=True, key=f"sel_{i}")
                elif len(st.session_state.selected_movies) < 5:
                    if st.button("➕ Select", key=f"sel_btn_{i}"):
                        st.session_state.selected_movies.append(movie_idx)
                        st.rerun()

st.subheader(f"🎥 Selected Movies ({len(st.session_state.selected_movies)}/5)")
st.markdown("<hr>", unsafe_allow_html=True)
if st.session_state.selected_movies:
    cols = st.columns(5)
    for i, movie_idx in enumerate(st.session_state.selected_movies):
        movie_row = df.loc[movie_idx]
        poster_url = fetch_poster(movie_row["poster_path"])
        with cols[i % 5]:
            with st.container():
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(poster_url, width=150)
                st.markdown(f'<div class="movie-title">{movie_row["original_title"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if st.button("❌ Remove", key=f"rem_{i}"):
                    st.session_state.selected_movies.remove(movie_idx)
                    st.rerun()
else:
    st.info("No movies selected yet.")

st.markdown("<br>", unsafe_allow_html=True)
if len(st.session_state.selected_movies) >= 3:
    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            rec_indices = get_recommendations(st.session_state.selected_movies, st.session_state.nn_model, feature_matrix)
            if rec_indices:
                st.subheader("🎬 Recommended Movies for You:")
                st.markdown("<hr>", unsafe_allow_html=True)
                cols = st.columns(5)
                for i, movie_idx in enumerate(rec_indices[:10]):
                    movie_row = df.iloc[movie_idx]
                    poster_url = fetch_poster(movie_row["poster_path"])
                    movie_id = movie_row["id"]
                    with cols[i % 5]:
                        with st.container():
                            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                            st.image(poster_url, width=150)
                            st.markdown(f'<div class="movie-title">{movie_row["original_title"]}</div>', unsafe_allow_html=True)
                            st.markdown(f'<a href="https://www.themoviedb.org/movie/{movie_id}" target="_blank" class="tmdb-button">View on TMDB</a>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please select at least 3 movies to get recommendations.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888;">Built using TMDB dataset and KNN with Cosine similarity.</div>', unsafe_allow_html=True)