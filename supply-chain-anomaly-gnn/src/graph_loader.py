"""Utility functions to load supply chain graph data."""

from typing import Tuple
import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


def load_graph(node_path: str, edge_path: str) -> Tuple[nx.Graph, Data]:
    """Load nodes and edges from CSV files and build graph objects.

    Parameters
    ----------
    node_path : str
        Path to the CSV file containing node information.
    edge_path : str
        Path to the CSV file containing edge information.

    Returns
    -------
    Tuple[nx.Graph, Data]
        A tuple containing the NetworkX graph and the PyTorch Geometric Data
        representation of the graph.
    """

    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edge_path)

    # Map textual node types to integer indices for use as features
    type_mapping = {t: i for i, t in enumerate(nodes_df["type"].unique())}

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        type_idx = type_mapping[row["type"]]
        risk_score = float(row["risk_score"])

        # Store original attributes
        G.add_node(
            node_id,
            type=row["type"],
            location=row["location"],
            risk_score=risk_score,
            x=torch.tensor([type_idx, risk_score], dtype=torch.float),
        )

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=float(row["weight"]),
            delay=float(row["delay"]),
        )

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G)

    return G, data
