#!/usr/bin/env python3
"""
run_gnn.py

Demonstration and inference script for the trained Graph Convolutional Network (GCN)
on the large-scale IoT scheduling dataset.

Loads:
  - Checkpoint: best_large_gcn_model.pt
  - Dataset:    large_dataset.pt

Selects a held-out test graph (default: synth_0099), runs slot-by-slot GNN inference,
compares predicted schedules against actual OR-Tools ground truth, displays scheduling
statistics and validity metrics, and produces a visual comparison plot.
"""

import argparse
import random
import sys
import warnings
from pathlib import Path

# Suppress PyTorch 3.14 future warnings for clean output
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
OUTPUT_PLOT = SCRIPT_DIR / "current_gnn_prediction.png"
ALT_OUTPUT_PLOT = SCRIPT_DIR / "gnn_prediction_visualization.png"

CLASS_NAMES = ["OFF", "TAG", "CG"]
ACTION_COLORS = {
    0: "#B0BEC5",  # OFF: Cool Slate Gray
    1: "#1E88E5",  # TAG: Vibrant Dodger Blue
    2: "#E53935",  # CG: Vivid Coral Red
}


# ==============================================================================
# Model Architecture (identical to train_large_gnn.py)
# ==============================================================================
class BaselineGCN(nn.Module):
    """
    2-layer Graph Convolutional Network.
    Input: [num_nodes, 5] (degree, hosted_tags, x, y, slot_fraction)
    Output: [num_nodes, 3] (logits for 0=OFF, 1=TAG, 2=CG)
    """

    def __init__(self, in_channels=5, hidden_channels=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.out(x)
        return logits


# ==============================================================================
# Graph-Level Split (Seed=42, 70% Train, 15% Val, 15% Test)
# ==============================================================================
def get_test_graph_indices(dataset, seed=42):
    """Replicate exact graph-level split from train_large_gnn.py."""
    random.seed(seed)
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    random.shuffle(indices)

    n_train = int(num_graphs * 0.70)
    n_val = int(num_graphs * 0.15)
    test_graph_indices = indices[n_train + n_val :]
    return test_graph_indices


# ==============================================================================
# Visual Plotting Function
# ==============================================================================
def generate_schedule_plot(sample, actual_matrix, pred_matrix, output_path, alt_output_path=None):
    """
    Renders a multi-slot side-by-side visualization of Actual vs Predicted schedules.
    """
    num_nodes = sample["num_nodes"]
    num_slots = sample["num_slots"]
    graph_id = sample["graph_id"]
    edge_index = sample["edge_index"]
    positions = sample.get("positions", None)

    # Build NetworkX graph
    G = nx.Graph()
    for n in range(num_nodes):
        G.add_node(n)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u < v:
            G.add_edge(u, v)

    # Node positions
    if positions is not None and positions.shape[0] == num_nodes:
        pos_dict = {n: (positions[n, 0].item(), positions[n, 1].item()) for n in range(num_nodes)}
    else:
        pos_dict = nx.spring_layout(G, seed=42)

    fig, axes = plt.subplots(num_slots, 2, figsize=(13, 4.5 * num_slots), squeeze=False)

    for s in range(num_slots):
        # Actual Subplot
        ax_act = axes[s][0]
        act_colors = [ACTION_COLORS[actual_matrix[n][s]] for n in range(num_nodes)]
        nx.draw_networkx_edges(G, pos_dict, ax=ax_act, edge_color="#90A4AE", width=1.5, alpha=0.7)
        nx.draw_networkx_nodes(
            G, pos_dict, ax=ax_act, node_color=act_colors, node_size=650,
            edgecolors="#263238", linewidths=1.5
        )
        labels_act = {n: f"{n}\n{CLASS_NAMES[actual_matrix[n][s]]}" for n in range(num_nodes)}
        nx.draw_networkx_labels(
            G, pos_dict, labels=labels_act, ax=ax_act, font_size=8,
            font_color="#FFFFFF", font_weight="bold"
        )
        ax_act.set_title(f"Slot {s}: Actual OR-Tools Schedule", fontsize=11, fontweight="bold")
        ax_act.axis("off")

        # Predicted Subplot
        ax_pred = axes[s][1]
        pred_colors = [ACTION_COLORS[pred_matrix[n][s]] for n in range(num_nodes)]
        nx.draw_networkx_edges(G, pos_dict, ax=ax_pred, edge_color="#90A4AE", width=1.5, alpha=0.7)
        nx.draw_networkx_nodes(
            G, pos_dict, ax=ax_pred, node_color=pred_colors, node_size=650,
            edgecolors="#263238", linewidths=1.5
        )
        labels_pred = {n: f"{n}\n{CLASS_NAMES[pred_matrix[n][s]]}" for n in range(num_nodes)}
        nx.draw_networkx_labels(
            G, pos_dict, labels=labels_pred, ax=ax_pred, font_size=8,
            font_color="#FFFFFF", font_weight="bold"
        )
        ax_pred.set_title(f"Slot {s}: GNN Predicted Schedule", fontsize=11, fontweight="bold")
        ax_pred.axis("off")

    legend_patches = [
        mpatches.Patch(color=ACTION_COLORS[0], label="OFF (Idle)"),
        mpatches.Patch(color=ACTION_COLORS[1], label="TAG (Interrogate)"),
        mpatches.Patch(color=ACTION_COLORS[2], label="CG (Carrier Gen)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11, frameon=True)
    fig.suptitle(
        f"GNN Schedule Inference vs Ground Truth\nGraph: {graph_id} | Nodes: {num_nodes} | Tags: {sample['num_tags']} | Slots: {num_slots}",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_file = Path(output_path)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    if alt_output_path:
        plt.savefig(Path(alt_output_path), dpi=200, bbox_inches="tight")
    plt.close()
    return out_file.name


# ==============================================================================
# Main Demonstration Runner
# ==============================================================================
def run_gnn_demo(
    dataset_path=DATASET_PATH,
    model_path=MODEL_PATH,
    target_graph_id="synth_0099",
    create_plot=True,
    output_plot=OUTPUT_PLOT,
):
    # 1. Validation of required files
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    # 2. Load dataset
    dataset = torch.load(dataset_path, weights_only=False)

    # 3. Obtain test split
    test_indices = get_test_graph_indices(dataset, seed=42)

    # 4. Select held-out test graph
    selected_sample = None
    for gi in test_indices:
        if dataset[gi]["graph_id"] == target_graph_id:
            selected_sample = dataset[gi]
            break

    if selected_sample is None:
        # Fallback to the first test graph if preferred ID is not present
        selected_sample = dataset[test_indices[0]]

    graph_id = selected_sample["graph_id"]
    num_nodes = selected_sample["num_nodes"]
    num_tags = selected_sample["num_tags"]
    num_slots = selected_sample["num_slots"]
    schedule_tensor = selected_sample["schedule"]  # [num_slots, num_nodes]

    # Print Selected Graph Info
    print("Selected test graph:")
    print(f"Graph ID: {graph_id}")
    print()
    print("Number of nodes:")
    print(num_nodes)
    print()
    print("Number of tags:")
    print(num_tags)
    print()
    print("Number of schedule slots:")
    print(num_slots)
    print()

    # 5. Load model and checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    input_dim = checkpoint.get("input_dim", 5)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    num_classes = checkpoint.get("num_classes", 3)
    dropout = checkpoint.get("dropout", 0.2)
    norm_mean = checkpoint["norm_mean"]
    norm_std = checkpoint["norm_std"]

    model = BaselineGCN(
        in_channels=input_dim,
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 6. Run Inference slot-by-slot
    pred_schedule_by_slot = []  # list of length num_slots, each [num_nodes]
    with torch.no_grad():
        for s in range(num_slots):
            slot_fraction = float(s) / max(1.0, float(num_slots - 1))
            fraction_col = torch.full((num_nodes, 1), slot_fraction, dtype=torch.float32)
            raw_x = torch.cat([selected_sample["node_features"], fraction_col], dim=-1)
            norm_x = (raw_x - norm_mean) / norm_std
            logits = model(norm_x, selected_sample["edge_index"], selected_sample["edge_weight"])
            slot_preds = logits.argmax(dim=-1).tolist()
            pred_schedule_by_slot.append(slot_preds)

    # Transpose to node-major matrices: [num_nodes][num_slots]
    actual_matrix = []
    pred_matrix = []
    for n in range(num_nodes):
        actual_matrix.append([schedule_tensor[s, n].item() for s in range(num_slots)])
        pred_matrix.append([pred_schedule_by_slot[s][n] for s in range(num_slots)])

    # 7. Print Actual Schedule
    print("============================================================")
    print("ACTUAL OR-TOOLS SCHEDULE")
    print("============================================================")
    print()
    header_str = "          " + "   ".join([f"Slot {s}" for s in range(num_slots)])
    print(header_str)
    for n in range(num_nodes):
        row_str = "   ".join([f"{CLASS_NAMES[actual_matrix[n][s]]:^6}" for s in range(num_slots)])
        print(f"Node {n:<4}  {row_str}")
    print()

    # 8. Print Predicted Schedule
    print("============================================================")
    print("GNN PREDICTED SCHEDULE")
    print("============================================================")
    print()
    print(header_str)
    for n in range(num_nodes):
        row_str = "   ".join([f"{CLASS_NAMES[pred_matrix[n][s]]:^6}" for s in range(num_slots)])
        print(f"Node {n:<4}  {row_str}")
    print()

    # 9. Compare Results
    total_decisions = num_nodes * num_slots
    correct_predictions = 0
    actual_counts = {0: 0, 1: 0, 2: 0}
    pred_counts = {0: 0, 1: 0, 2: 0}

    print("============================================================")
    print("SCHEDULE COMPARISON")
    print("============================================================")
    print()
    print("Per-Node Decision Comparison:")
    for n in range(num_nodes):
        for s in range(num_slots):
            act = actual_matrix[n][s]
            prd = pred_matrix[n][s]
            actual_counts[act] += 1
            pred_counts[prd] += 1
            match = (act == prd)
            if match:
                correct_predictions += 1
            status = "MATCH" if match else "MISMATCH"
            print(f"Node {n:<2} Slot {s}: Actual = {CLASS_NAMES[act]:<3} | Predicted = {CLASS_NAMES[prd]:<3} [{status}]")

    incorrect_predictions = total_decisions - correct_predictions
    accuracy = correct_predictions / total_decisions

    print()
    print(f"Total node-slot decisions: {total_decisions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Schedule classification accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    print("Actual:")
    print(f"OFF = {actual_counts[0]}")
    print(f"TAG = {actual_counts[1]}")
    print(f"CG = {actual_counts[2]}")
    print()
    print("Predicted:")
    print(f"OFF = {pred_counts[0]}")
    print(f"TAG = {pred_counts[1]}")
    print(f"CG = {pred_counts[2]}")
    print()

    # 10. TAG / CG Analysis
    print("============================================================")
    print("TAG / CG ANALYSIS")
    print("============================================================")
    print()
    print(f"Actual TAG count: {actual_counts[1]}")
    print(f"Predicted TAG count: {pred_counts[1]}")
    print()
    print(f"Actual CG count: {actual_counts[2]}")
    print(f"Predicted CG count: {pred_counts[2]}")
    print()
    print(f"Actual OFF count: {actual_counts[0]}")
    print(f"Predicted OFF count: {pred_counts[0]}")
    print()
    print("Hosted tags:")
    print(num_tags)
    print()
    print("Predicted TAG actions:")
    print(pred_counts[1])
    print()
    diff = pred_counts[1] - num_tags
    print("Difference:")
    print(f"{diff:+d}")
    print()

    # 11. Model Limitation Message
    print("Note: The GCN predicts each schedule slot independently. Therefore, the predicted schedule is not guaranteed to satisfy all global OR-Tools scheduling constraints such as exact tag interrogation counts and carrier compatibility.")
    print()

    # 12. Visualization
    vis_file = "None"
    if create_plot:
        vis_file = generate_schedule_plot(
            selected_sample, actual_matrix, pred_matrix, output_plot, ALT_OUTPUT_PLOT
        )

    # 13. Final Terminal Output
    print("============================================================")
    print("GNN INFERENCE COMPLETE")
    print("============================================================")
    print()
    print("Model:")
    print(Path(model_path).name)
    print()
    print("Test graph:")
    print(graph_id)
    print()
    print("Nodes:")
    print(num_nodes)
    print()
    print("Tags:")
    print(num_tags)
    print()
    print("Slots:")
    print(num_slots)
    print()
    print("Classification accuracy:")
    print(f"{accuracy * 100:.2f}%")
    print()
    print("Actual TAG:")
    print(actual_counts[1])
    print()
    print("Predicted TAG:")
    print(pred_counts[1])
    print()
    print("Actual CG:")
    print(actual_counts[2])
    print()
    print("Predicted CG:")
    print(pred_counts[2])
    print()
    print("Actual OFF:")
    print(actual_counts[0])
    print()
    print("Predicted OFF:")
    print(pred_counts[0])
    print()
    print("Visualization:")
    print(vis_file)
    print()
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate GCN inference on large IoT dataset.")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH), help="Path to large_dataset.pt")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to best_large_gcn_model.pt")
    parser.add_argument("--graph-id", default="synth_0099", help="Target test graph ID (default: synth_0099)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization PNG")
    parser.add_argument("--output-plot", default=str(OUTPUT_PLOT), help="Output visualization file path")
    args = parser.parse_args()

    run_gnn_demo(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        target_graph_id=args.graph_id,
        create_plot=not args.no_plot,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
