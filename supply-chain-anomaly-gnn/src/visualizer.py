"""Graph visualization utilities for supply chain anomalies."""

from typing import Iterable
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(graph: nx.Graph, anomalies: Iterable[str]):
    """Plot the supply chain graph highlighting anomalous nodes."""

    pos = nx.spring_layout(graph)
    anomaly_set = set(anomalies)

    node_colors = ["red" if n in anomaly_set else "skyblue" for n in graph.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    plt.title("Supply Chain Network with Anomalies")
    plt.axis("off")
    return plt.gcf()
