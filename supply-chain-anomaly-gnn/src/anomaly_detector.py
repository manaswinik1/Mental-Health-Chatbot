"""Detect anomalies in supply chain graphs using node embeddings."""

from typing import Dict
import networkx as nx
import torch


def detect_anomalies(graph: nx.Graph, embeddings: torch.Tensor, threshold: float = 2.0) -> Dict[str, float]:
    """Identify anomalous nodes based on embedding distance from the mean.

    Parameters
    ----------
    graph : nx.Graph
        Original NetworkX graph with node identifiers.
    embeddings : torch.Tensor
        Node embeddings produced by a GNN; order must correspond to graph nodes.
    threshold : float, optional
        Z-score threshold to flag anomalies, by default 2.0.

    Returns
    -------
    Dict[str, float]
        Mapping of node identifiers to anomaly scores (z-scores).
    """

    nodes = list(graph.nodes())
    mean = embeddings.mean(dim=0)
    distances = torch.norm(embeddings - mean, dim=1)

    z_scores = (distances - distances.mean()) / (distances.std() + 1e-9)

    anomalies = {
        nodes[i]: z_scores[i].item()
        for i in range(len(nodes))
        if z_scores[i].item() > threshold
    }

    return anomalies
