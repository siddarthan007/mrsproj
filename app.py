import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

@st.cache_data
def load_data():
    try:
        save_dict = joblib.load("models/recommender_model.joblib")  
        return save_dict['df'], save_dict['feature_matrix'], save_dict['knn_model'], save_dict['movie_ids']  
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

df, feature_matrix, knn_model, movie_ids = load_data()
if df is None or feature_matrix is None or knn_model is None:
    st.stop()

def fetch_poster(poster_path):
    if not poster_path or pd.isna(poster_path):
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

@st.cache_data
def get_recommendations(selected_indices, _nn_model, matrix, n_recommend=10):
    try:
        distances, indices = _nn_model.kneighbors(matrix[selected_indices])
        score_dict = {}
        for dist, idx in zip(distances, indices):
            for d, i in zip(dist, idx):
                if i not in selected_indices:
                    similarity = 1 - d
                    score_dict[i] = score_dict.get(i, 0) + similarity
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        rec_indices = [item[0] for item in sorted_scores[:n_recommend]]
        rec_scores = [item[1] for item in sorted_scores[:n_recommend]]
        return rec_indices, rec_scores
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], []

st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 36px; font-weight: bold; color: white; margin-bottom: 10px; }
        .sub-title { text-align: center; font-size: 18px; color: white; margin-bottom: 20px; }
        .movie-container { display: flex; flex-direction: column; align-items: center; text-align: center; margin-bottom: 20px; }
        .movie-title { font-size: 14px; font-weight: bold; margin-top: 10px; width: 150px; color: white; }
        .genres { font-size: 12px; color: #CCCCCC; margin-top: 2px; }
        .button { background-color: #3498DB; color: white; border: none; padding: 8px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; margin: 4px 2px; cursor: pointer; border-radius: 4px; }
        .button:hover { background-color: #2980B9; }
        hr { border: 0; height: 1px; background: #BDC3C7; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Select up to 5 movies to get personalized recommendations</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

advanced_insights = st.checkbox("Enable Advanced Insights", value=False)

if 'selected_movies' not in st.session_state:
    st.session_state['selected_movies'] = []

search_query = st.text_input("Search for a movie:", placeholder="Type a movie name...")
filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)].head(20) if search_query else pd.DataFrame()

if not filtered_df.empty:
    st.subheader("Search Results")
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
                    st.button("Selected", disabled=True, key=f"sel_{i}", help="This movie is already selected.")
                elif len(st.session_state['selected_movies']) < 5:
                    if st.button("Select", key=f"sel_btn_{i}", help="Add this movie to your selection."):
                        st.session_state['selected_movies'].append(movie_id)
                        st.rerun()

st.subheader(f"Selected Movies ({len(st.session_state['selected_movies'])}/5)")
st.markdown("<hr>", unsafe_allow_html=True)

if st.session_state['selected_movies'] and st.button("Clear All", help="Remove all selected movies."):
    st.session_state['selected_movies'] = []
    st.rerun()

if st.session_state['selected_movies']:
    cols = st.columns(min(len(st.session_state['selected_movies']), 5))
    for i, movie_id in enumerate(st.session_state['selected_movies']):
        movie = df[df['movieId'] == movie_id].iloc[0]
        poster_url = fetch_poster(movie['poster_path'])
        with cols[i % 5]:
            with st.container():
                st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                st.image(poster_url, width=150)
                st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="genres">{", ".join(movie["genres"])}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if st.button("Remove", key=f"rem_{i}", help="Remove this movie from your selection."):
                    st.session_state['selected_movies'].remove(movie_id)
                    st.rerun()
else:
    st.info("No movies selected yet.")

if len(st.session_state['selected_movies']) >= 3:
    if st.button("Get Recommendations", use_container_width=True, help="Generate movie recommendations based on your selection."):
        with st.spinner("Generating recommendations..."):
            selected_indices = [np.where(movie_ids == mid)[0][0] for mid in st.session_state['selected_movies']]
            rec_indices, rec_scores = get_recommendations(selected_indices, knn_model, feature_matrix)
            if rec_indices:
                st.subheader("Recommended Movies for You")
                st.markdown("<hr>", unsafe_allow_html=True)
                cols = st.columns(5)
                for i, (idx, score) in enumerate(zip(rec_indices[:10], rec_scores[:10])):
                    movie = df.iloc[idx]
                    poster_url = fetch_poster(movie['poster_path'])
                    with cols[i % 5]:
                        with st.container():
                            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
                            st.image(poster_url, width=150)
                            st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="font-size: 12px; color: #888;">Score: {score:.2f}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please select at least 3 movies to get recommendations.")

if advanced_insights and st.session_state['selected_movies']:
    with st.sidebar:
        st.subheader("Advanced Insights")
        selected_indices = [np.where(movie_ids == mid)[0][0] for mid in st.session_state['selected_movies']]
        rec_indices, rec_scores = get_recommendations(selected_indices, knn_model, feature_matrix, n_recommend=10)

        with st.expander("Visualizations"):
            if len(rec_indices) > 1:
                rec_features = feature_matrix[rec_indices]
                sim_matrix = cosine_similarity(rec_features)

                if st.checkbox("Visualize Feature Space", value=False):
                    with st.spinner("Generating 2D projection..."):
                        combined_indices = selected_indices + rec_indices
                        relevant_features = feature_matrix[combined_indices]
                        reducer = umap.UMAP(random_state=42)
                        embedding = reducer.fit_transform(relevant_features)
                        num_selected = len(selected_indices)
                        selected_embedding = embedding[:num_selected]
                        recommended_embedding = embedding[num_selected:]

                        df_plot = pd.DataFrame({
                            'x': np.concatenate([selected_embedding[:, 0], recommended_embedding[:, 0]]),
                            'y': np.concatenate([selected_embedding[:, 1], recommended_embedding[:, 1]]),
                            'Category': ['Selected'] * num_selected + ['Recommended'] * len(recommended_embedding),
                            'Title': [df.iloc[i]['title'] for i in combined_indices]
                        })

                        fig = px.scatter(
                            df_plot,
                            x='x',
                            y='y',
                            color='Category',
                            hover_data=['Title'],
                            color_discrete_map={"Selected": "red", "Recommended": "blue"},
                            title="Feature Space"
                        )

                        fig.update_layout(
                            xaxis_title="X",
                            yaxis_title="Y",
                            legend_title="Movie Category",
                            margin=dict(l=20, r=20, t=50, b=20),
                            template="plotly_white",
                            hovermode="closest"
                        )

                        fig.update_traces(
                            hovertemplate="<b>%{customdata[0]}</b><br>" +
                                        "X: %{x:.2f}<br>" +
                                        "Y: %{y:.2f}<extra></extra>"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                if st.checkbox("Show Similarity Heatmap", value=False):
                    sim_matrix = cosine_similarity(feature_matrix[selected_indices], feature_matrix[rec_indices])
                    fig, ax = plt.subplots()
                    sns.heatmap(sim_matrix, annot=True, cmap="YlGnBu", ax=ax)
                    ax.set_xticklabels([df.iloc[idx]['title'][:15] + '...' for idx in rec_indices], rotation=45, ha='right')
                    ax.set_yticklabels([df.iloc[idx]['title'][:15] + '...' for idx in selected_indices], rotation=0)
                    st.pyplot(fig)

        with st.expander("Recommendation Scores"):
            if rec_indices:
                rec_movies = df.iloc[rec_indices][['title']]
                rec_movies['Total Similarity'] = rec_scores
                rec_movies['Average Similarity'] = rec_movies['Total Similarity'] / len(selected_indices)
                st.write("**Recommended Movies with Similarity Scores:**")
                st.dataframe(rec_movies.style.format({"Total Similarity": "{:.4f}", "Average Similarity": "{:.4f}"}))
                avg_rec_similarity = np.mean(rec_movies['Average Similarity'])
                st.write(f"**Average similarity of recommendations:** {avg_rec_similarity:.4f}")
            else:
                st.write("No recommendations generated yet.")

        with st.expander("Inspect Recommended Movie"):
            if rec_indices:
                rec_titles = [df.iloc[idx]['title'] for idx in rec_indices]
                selected_rec = st.selectbox("Select a recommended movie:", options=rec_titles)
                rec_idx = rec_indices[rec_titles.index(selected_rec)]
                rec_feature = feature_matrix[rec_idx].reshape(1, -1)
                similarities = cosine_similarity(rec_feature, feature_matrix[selected_indices])[0]
                sim_df = pd.DataFrame({
                    "Selected Movie": [df.iloc[idx]['title'] for idx in selected_indices],
                    "Similarity": similarities
                })
                sim_df = sim_df.sort_values(by="Similarity", ascending=False)
                st.write(f"**Similarities to {selected_rec}:**")
                st.dataframe(sim_df.style.format({"Similarity": "{:.4f}"}))
            else:
                st.write("No recommendations to inspect.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888;">Built using MovieLens dataset and trained using K-Nearest Neighbors with Cosine similarity as metric.</div>', unsafe_allow_html=True)