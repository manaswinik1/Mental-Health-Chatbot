import streamlit as st

from src.data_loader import load_data
from src.hybrid_recommender import (
    hybrid_recommend,
    _compute_cf_scores,
    _compute_content_scores,
)
from src.content_embedder import generate_content_embeddings
from src.explainer import explain_recommendation

ratings_df, metadata_df = load_data()
embeddings = generate_content_embeddings(metadata_df)

st.title("Explainable Hybrid Recommendation System")

user_id = st.selectbox("Select User ID", sorted(ratings_df["user_id"].unique()))

if st.button("Get Recommendations"):
    recommendations = hybrid_recommend(user_id)
    cf_scores = _compute_cf_scores(user_id, ratings_df)
    content_scores = _compute_content_scores(user_id, ratings_df, metadata_df, embeddings)

    st.subheader("Top Recommendations")
    for item_id in recommendations:
        item = metadata_df[metadata_df["item_id"] == item_id].iloc[0]
        st.markdown(f"### {item['title']}")
        st.write(item["description"])
        cf_score = cf_scores.get(item_id, 0.0)
        content_score = content_scores.get(item_id, 0.0)
        if st.button("Why?", key=f"exp-{item_id}"):
            explanation = explain_recommendation(cf_score, content_score)
            st.info(explanation)
