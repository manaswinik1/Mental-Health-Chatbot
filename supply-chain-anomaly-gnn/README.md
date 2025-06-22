# Supply Chain Anomaly Detection Using GNNs

This project demonstrates how Graph Neural Networks (GNNs) can be used to model
supply chain networks and detect structural anomalies, bottlenecks, or risky
nodes. It is a small, self-contained example using synthetic data and a
Streamlit dashboard for visualization.

## Key Features
- Graph-based representation of supply chain nodes and edges
- Node embeddings generated with a simple GCN model
- Basic anomaly detection using distances in embedding space
- Interactive Streamlit dashboard to visualize results

## Dataset
The data in `data/raw/` is synthetic and serves as a demonstration. It consists
of two CSV files:

- `supply_chain_nodes.csv` — node ID, type, location, and risk score
- `supply_chain_edges.csv` — source/target pairs with weights and delays

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app/streamlit_dashboard.py
   ```

## Folder Structure
```
supply-chain-anomaly-gnn/
├── data/
│   └── raw/
│       ├── supply_chain_edges.csv
│       └── supply_chain_nodes.csv
├── models/
├── src/
│   ├── anomaly_detector.py
│   ├── gnn_model.py
│   ├── graph_loader.py
│   └── visualizer.py
├── app/
│   └── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

![Dashboard Screenshot](docs/screenshot.png)

*Disclaimer: This project uses simulated data and does not represent real-world
supply chain intelligence.*
