"""Streamlit dashboard for visualizing supply chain anomalies."""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.graph_loader import load_graph
from src.gnn_model import train_gnn
from src.anomaly_detector import detect_anomalies
from src.visualizer import plot_graph


def main():
    st.title("Supply Chain Anomaly Detection Using GNNs")

    default_node = "data/raw/supply_chain_nodes.csv"
    default_edge = "data/raw/supply_chain_edges.csv"

    node_path = st.sidebar.text_input("Node CSV", default_node)
    edge_path = st.sidebar.text_input("Edge CSV", default_edge)

    risk_filter = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.0, 0.05)

    if st.sidebar.button("Run Analysis"):
        graph, data = load_graph(node_path, edge_path)
        with st.spinner("Training GNN ..."):
            _, embeddings = train_gnn(data)
        anomalies = detect_anomalies(graph, embeddings)

        st.subheader("Anomaly Scores")
        st.write(anomalies)

        filtered_nodes = [n for n in anomalies if graph.nodes[n]["risk_score"] >= risk_filter]

        fig = plot_graph(graph, filtered_nodes)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
