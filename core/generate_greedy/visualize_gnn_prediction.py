#!/usr/bin/env python3
"""
visualize_gnn_prediction.py

Visualizes a test topology with actual vs. predicted scheduling actions from
best_large_gcn_model.pt.
- Side-by-side plot: Actual vs. Predicted node actions for a schedule slot
- Shows IoT nodes, communication edges, node IDs, and action labels
- Colors: OFF = Light Gray, TAG = Dodger Blue, CG = Coral Red
"""

import argparse
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "large_dataset.pt"
MODEL_PATH = SCRIPT_DIR / "best_large_gcn_model.pt"
OUTPUT_IMAGE = SCRIPT_DIR / "gnn_prediction_visualization.png"

CLASS_NAMES = ["OFF", "TAG", "CG"]
ACTION_COLORS = {
    0: "#B0BEC5",  # OFF: Cool Gray
    1: "#1E88E5",  # TAG: Blue
    2: "#E53935",  # CG: Red
}


class BaselineGCN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        return self.out(x)


def visualize_test_graph(
    dataset_path=DATASET_PATH,
    model_path=MODEL_PATH,
    output_path=OUTPUT_IMAGE,
    slot_idx=0,
    seed=42,
):
    if not Path(dataset_path).exists():
        print(f"Dataset not found at: {dataset_path}")
        return False
    if not Path(model_path).exists():
        print(f"Model checkpoint not found at: {model_path}. Train the model first.")
        return False

    # 1. Load dataset & reproduce test split
    dataset = torch.load(dataset_path, weights_only=False)
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_train = int(len(dataset) * 0.70)
    n_val = int(len(dataset) * 0.15)
    test_indices = indices[n_train + n_val :]

    # 2. Select a representative test graph
    selected = None
    for gi in test_indices:
        s = dataset[gi]
        if s["num_slots"] >= 2 and 6 <= s["num_nodes"] <= 12:
            selected = s
            break
    if selected is None:
        selected = dataset[test_indices[0]]

    graph_id = selected["graph_id"]
    num_nodes = selected["num_nodes"]
    num_slots = selected["num_slots"]
    edge_index = selected["edge_index"]
    edge_weight = selected["edge_weight"]
    positions = selected.get("positions", None)
    schedule = selected["schedule"]  # [num_slots, num_nodes]

    # 3. Load checkpoint & model
    checkpoint = torch.load(model_path, weights_only=False)
    norm_mean = checkpoint["norm_mean"]
    norm_std = checkpoint["norm_std"]

    model = BaselineGCN(in_channels=5, hidden_channels=64, num_classes=3, dropout=0.2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 4. Predict for slot_idx
    slot = min(slot_idx, num_slots - 1)
    slot_fraction = float(slot) / max(1.0, float(num_slots - 1))
    fraction_col = torch.full((num_nodes, 1), slot_fraction, dtype=torch.float32)
    raw_x = torch.cat([selected["node_features"], fraction_col], dim=-1)
    norm_x = (raw_x - norm_mean) / norm_std

    with torch.no_grad():
        logits = model(norm_x, edge_index, edge_weight)
        pred_actions = logits.argmax(dim=-1).tolist()

    actual_actions = schedule[slot].tolist()

    # 5. Build NetworkX Graph
    G = nx.Graph()
    for n in range(num_nodes):
        G.add_node(n)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u < v:
            G.add_edge(u, v)

    # Positions
    if positions is not None and positions.shape[0] == num_nodes:
        pos_dict = {n: (positions[n, 0].item(), positions[n, 1].item()) for n in range(num_nodes)}
    else:
        pos_dict = nx.spring_layout(G, seed=seed)

    # 6. Plot Side-by-Side: Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Actual Subplot
    ax_act = axes[0]
    act_colors = [ACTION_COLORS[actual_actions[n]] for n in range(num_nodes)]
    nx.draw_networkx_edges(G, pos_dict, ax=ax_act, edge_color="#90A4AE", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos_dict, ax=ax_act, node_color=act_colors, node_size=600, edgecolors="#37474F", linewidths=1.5)
    labels_act = {n: f"{n}\n({CLASS_NAMES[actual_actions[n]]})" for n in range(num_nodes)}
    nx.draw_networkx_labels(G, pos_dict, labels=labels_act, ax=ax_act, font_size=8, font_color="#FFFFFF", font_weight="bold")
    ax_act.set_title(f"Actual Schedule (Slot {slot})\nGraph: {graph_id} ({num_nodes} nodes, {selected['num_tags']} tags)", fontsize=12, fontweight="bold")
    ax_act.axis("off")

    # Predicted Subplot
    ax_pred = axes[1]
    pred_colors = [ACTION_COLORS[pred_actions[n]] for n in range(num_nodes)]
    nx.draw_networkx_edges(G, pos_dict, ax=ax_pred, edge_color="#90A4AE", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos_dict, ax=ax_pred, node_color=pred_colors, node_size=600, edgecolors="#37474F", linewidths=1.5)
    labels_pred = {n: f"{n}\n({CLASS_NAMES[pred_actions[n]]})" for n in range(num_nodes)}
    nx.draw_networkx_labels(G, pos_dict, labels=labels_pred, ax=ax_pred, font_size=8, font_color="#FFFFFF", font_weight="bold")
    ax_pred.set_title(f"GCN Predicted Schedule (Slot {slot})\nGraph: {graph_id}", fontsize=12, fontweight="bold")
    ax_pred.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=ACTION_COLORS[0], label="OFF (Idle)"),
        mpatches.Patch(color=ACTION_COLORS[1], label="TAG (Interrogate)"),
        mpatches.Patch(color=ACTION_COLORS[2], label="CG (Carrier Gen)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Visualization successfully saved to: {output_path.resolve()}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize GCN predictions vs actual schedule.")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH), help="Path to large_dataset.pt")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to best_large_gcn_model.pt")
    parser.add_argument("--output", default=str(OUTPUT_IMAGE), help="Path to output PNG image")
    parser.add_argument("--slot", type=int, default=0, help="Schedule slot to visualize (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split reproducibility (default: 42)")
    args = parser.parse_args()

    visualize_test_graph(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_path=args.output,
        slot_idx=args.slot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
