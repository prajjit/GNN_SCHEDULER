#!/usr/bin/env python3
"""
train_large_gnn.py

Trains a Graph Convolutional Network (GCN) on the large-scale IoT scheduling
dataset (large_dataset.pt).
- 500 graphs, split at graph-level: 70% train (350), 15% val (75), 15% test (75)
- Reproducible with SEED=42
- Node features (5-dim): [degree, hosted_tags, x, y, slot_fraction]
- Normalization computed strictly on the training set
- Balanced class weights computed strictly on the training set
- 2-layer GCN (in_channels=5, hidden_channels=64, num_classes=3)
- Early stopping based on validation macro-F1 (patience=15, max 100 epochs)
- Saves best checkpoint to best_large_gcn_model.pt and training history to large_gnn_training_history.json
- Evaluates on held-out test graphs with confusion matrix, per-class metrics, and distribution analysis
- Demonstrates actual vs predicted schedule on a test graph with scheduling validity analysis
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

# Suppress PyTorch 3.14 future warnings for clean terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch_geometric.nn import GCNConv

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "large_dataset.pt"
MODEL_SAVE_PATH = SCRIPT_DIR / "best_large_gcn_model.pt"
HISTORY_SAVE_PATH = SCRIPT_DIR / "large_gnn_training_history.json"

CLASS_NAMES = ["OFF", "TAG", "CG"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_slot_examples(dataset, graph_indices):
    """
    Decompose selected graphs into slot-level examples.
    For each graph and each schedule slot:
      slot_fraction = slot / max(1, num_slots - 1)
      node features = [degree, hosted_tags, x, y, slot_fraction]  (shape: [num_nodes, 5])
      target = schedule[slot]  (shape: [num_nodes])
    """
    examples = []
    for gi in graph_indices:
        sample = dataset[gi]
        num_nodes = sample["num_nodes"]
        num_slots = sample["num_slots"]
        base_x = sample["node_features"]  # [num_nodes, 4]
        schedule = sample["schedule"]  # [num_slots, num_nodes]
        edge_index = sample["edge_index"]
        edge_weight = sample["edge_weight"]
        graph_id = sample["graph_id"]

        for s in range(num_slots):
            slot_fraction = float(s) / max(1.0, float(num_slots - 1))
            fraction_col = torch.full((num_nodes, 1), slot_fraction, dtype=torch.float32)
            x_with_slot = torch.cat([base_x, fraction_col], dim=-1)  # [num_nodes, 5]
            y = schedule[s]  # [num_nodes]

            examples.append(
                {
                    "graph_id": graph_id,
                    "graph_idx": gi,
                    "slot": s,
                    "num_nodes": num_nodes,
                    "num_slots": num_slots,
                    "x": x_with_slot,
                    "edge_index": edge_index,
                    "edge_weight": edge_weight,
                    "y": y,
                }
            )
    return examples


def evaluate(model, examples, criterion):
    """Evaluate model on a list of slot examples."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ex in examples:
            logits = model(ex["x_norm"], ex["edge_index"], ex["edge_weight"])
            loss = criterion(logits, ex["y"])
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(ex["y"].tolist())

    avg_loss = total_loss / max(1, len(examples))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1, all_preds, all_targets


def train(epochs=100, lr=0.001, patience=15, seed=42, dataset_path=DATASET_PATH):
    set_seed(seed)

    # 1. Load dataset
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    dataset = torch.load(dataset_path, weights_only=False)
    num_graphs = len(dataset)
    total_nodes = sum(s["num_nodes"] for s in dataset)
    total_tags = sum(s["num_tags"] for s in dataset)
    total_slots = sum(s["num_slots"] for s in dataset)

    # Total class distribution across dataset
    dataset_actions = {0: 0, 1: 0, 2: 0}
    for s in dataset:
        sched = s["schedule"]
        for lbl in (0, 1, 2):
            dataset_actions[lbl] += (sched == lbl).sum().item()

    # 2. Graph-level Split (70% train, 15% val, 15% test)
    # For 500 graphs: 350 train, 75 val, 75 test
    n_train = int(num_graphs * 0.70)
    n_val = int(num_graphs * 0.15)
    n_test = num_graphs - n_train - n_val

    indices = list(range(num_graphs))
    random.shuffle(indices)

    train_graph_indices = indices[:n_train]
    val_graph_indices = indices[n_train : n_train + n_val]
    test_graph_indices = indices[n_train + n_val :]

    train_graph_ids = [dataset[i]["graph_id"] for i in train_graph_indices]
    val_graph_ids = [dataset[i]["graph_id"] for i in val_graph_indices]
    test_graph_ids = [dataset[i]["graph_id"] for i in test_graph_indices]

    # Build slot-level examples
    train_examples = build_slot_examples(dataset, train_graph_indices)
    val_examples = build_slot_examples(dataset, val_graph_indices)
    test_examples = build_slot_examples(dataset, test_graph_indices)

    # 3. Feature Normalization (TRAINING-SET STATISTICS ONLY)
    all_train_features = torch.cat([ex["x"] for ex in train_examples], dim=0)
    norm_mean = all_train_features.mean(dim=0)
    norm_std = all_train_features.std(dim=0)
    norm_std[norm_std < 1e-6] = 1.0  # Avoid division by zero

    for ex in train_examples + val_examples + test_examples:
        ex["x_norm"] = (ex["x"] - norm_mean) / norm_std

    # 4. Class Imbalance (TRAINING SET ONLY)
    train_targets = torch.cat([ex["y"] for ex in train_examples])
    train_class_counts = torch.bincount(train_targets, minlength=3).float()
    total_train_nodes = float(len(train_targets))

    # Balanced class weights: total / (num_classes * count)
    class_weights = total_train_nodes / (3.0 * train_class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Print setup summary
    print("============================================================")
    print("GCN TRAINING ON LARGE DATASET (500 GRAPHS)")
    print("============================================================")
    print()
    print("Dataset Overview:")
    print(f"Total Graphs: {num_graphs}")
    print(f"Total Active Nodes: {total_nodes}")
    print(f"Total Tags: {total_tags}")
    print(f"Total Schedule Slots: {total_slots}")
    print()
    print("Graph-Level Split (Seed=42):")
    print(f"Training graphs count:   {len(train_graph_indices)} (70%)")
    print(f"Validation graphs count: {len(val_graph_indices)} (15%)")
    print(f"Test graphs count:       {len(test_graph_indices)} (15%)")
    print()
    print("Concise Graph IDs:")
    print(f"Train IDs: [min={min(train_graph_ids)}, max={max(train_graph_ids)}, total={len(train_graph_ids)}]")
    print(f"Val IDs:   [min={min(val_graph_ids)}, max={max(val_graph_ids)}, total={len(val_graph_ids)}]")
    print(f"Test IDs:  [min={min(test_graph_ids)}, max={max(test_graph_ids)}, total={len(test_graph_ids)}]")
    print()
    print(f"Slot Examples: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    print()
    print("Class Distribution (Overall):")
    print(f"OFF: {dataset_actions[0]}")
    print(f"TAG: {dataset_actions[1]}")
    print(f"CG:  {dataset_actions[2]}")
    print()
    print("Training Set Class Weights (Balanced):")
    print(f"OFF class weight: {class_weights[0].item():.4f} (count={int(train_class_counts[0].item())})")
    print(f"TAG class weight: {class_weights[1].item():.4f} (count={int(train_class_counts[1].item())})")
    print(f"CG  class weight: {class_weights[2].item():.4f} (count={int(train_class_counts[2].item())})")
    print()
    print("Model Architecture:")
    print("Input: [num_nodes, 5] (degree, hosted_tags, x, y, slot_fraction)")
    print("Layer 1: GCNConv(in_channels=5, hidden_channels=64) -> ReLU -> Dropout(0.2)")
    print("Layer 2: GCNConv(in_channels=64, hidden_channels=64) -> ReLU -> Dropout(0.2)")
    print("Output : Linear(in_features=64, out_features=3) -> logits for [OFF, TAG, CG]")
    print()

    # 5. Initialize Model & Optimizer
    model = BaselineGCN(in_channels=5, hidden_channels=64, num_classes=3, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. Training Loop
    print("============================================================")
    print("STARTING TRAINING (Max Epochs: 100, Patience: 15)")
    print("============================================================")

    best_val_macro_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_preds = []
        train_targets_list = []

        # Shuffle training slot examples each epoch
        random.shuffle(train_examples)

        for ex in train_examples:
            optimizer.zero_grad()
            logits = model(ex["x_norm"], ex["edge_index"], ex["edge_weight"])
            loss = criterion(logits, ex["y"])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_preds.extend(logits.argmax(dim=-1).tolist())
            train_targets_list.extend(ex["y"].tolist())

        train_loss = epoch_loss / max(1, len(train_examples))
        train_acc = accuracy_score(train_targets_list, train_preds)

        # Evaluate Validation
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(model, val_examples, criterion)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        # Early stopping and model selection on validation macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print()
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best validation Macro F1 was {best_val_macro_f1:.4f} at epoch {best_epoch}."
                )
                break

    # 7. Save Best Model Checkpoint
    checkpoint = {
        "state_dict": best_model_state,
        "input_dim": 5,
        "hidden_dim": 64,
        "num_classes": 3,
        "dropout": 0.2,
        "norm_mean": norm_mean.cpu(),
        "norm_std": norm_std.cpu(),
        "class_weights": class_weights.cpu(),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print()
    print(f"Best model checkpoint saved to: {MODEL_SAVE_PATH.name}")

    # 8. Test Evaluation on Held-out Test Graphs
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_macro_f1, test_preds, test_targets = evaluate(
        model, test_examples, criterion
    )

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_preds, labels=[0, 1, 2], zero_division=0
    )

    # Confusion Matrix (Rows = Actual, Columns = Predicted)
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])

    # Distribution counts
    pred_counts = {0: 0, 1: 0, 2: 0}
    actual_counts = {0: 0, 1: 0, 2: 0}
    for p in test_preds:
        pred_counts[p] += 1
    for t in test_targets:
        actual_counts[t] += 1

    all_three_predicted = all(pred_counts[c] > 0 for c in (0, 1, 2))

    # Save training history JSON
    history_data = {
        "split": {
            "num_train_graphs": len(train_graph_ids),
            "num_val_graphs": len(val_graph_ids),
            "num_test_graphs": len(test_graph_ids),
            "train_graph_ids": train_graph_ids,
            "val_graph_ids": val_graph_ids,
            "test_graph_ids": test_graph_ids,
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_results": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(support[i]),
                }
                for i in range(3)
            },
            "confusion_matrix": cm.tolist(),
            "prediction_distribution": {CLASS_NAMES[k]: v for k, v in pred_counts.items()},
            "actual_distribution": {CLASS_NAMES[k]: v for k, v in actual_counts.items()},
            "all_three_classes_predicted": all_three_predicted,
        },
        "epochs": history,
    }

    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved to: {HISTORY_SAVE_PATH.name}")
    print()

    # 9. Select ONE Graph from Held-out Test Set for Demonstration
    # Pick a test graph with >=2 slots and reasonable node count for clear tabular review
    selected_sample = None
    selected_idx = None
    for gi in test_graph_indices:
        s = dataset[gi]
        if s["num_slots"] >= 2 and 6 <= s["num_nodes"] <= 12:
            selected_sample = s
            selected_idx = gi
            break
    if selected_sample is None:
        # Fallback to the first test graph
        selected_idx = test_graph_indices[0]
        selected_sample = dataset[selected_idx]

    sel_graph_id = selected_sample["graph_id"]
    sel_num_nodes = selected_sample["num_nodes"]
    sel_num_tags = selected_sample["num_tags"]
    sel_num_slots = selected_sample["num_slots"]
    sel_schedule = selected_sample["schedule"]  # [num_slots, num_nodes]

    # Predict schedule for selected graph slot-by-slot
    model.eval()
    pred_schedule_matrix = []
    with torch.no_grad():
        for s in range(sel_num_slots):
            slot_fraction = float(s) / max(1.0, float(sel_num_slots - 1))
            fraction_col = torch.full((sel_num_nodes, 1), slot_fraction, dtype=torch.float32)
            raw_x = torch.cat([selected_sample["node_features"], fraction_col], dim=-1)
            norm_x = (raw_x - norm_mean) / norm_std
            logits = model(norm_x, selected_sample["edge_index"], selected_sample["edge_weight"])
            slot_preds = logits.argmax(dim=-1).tolist()
            pred_schedule_matrix.append(slot_preds)

    pred_schedule_matrix = list(map(list, zip(*pred_schedule_matrix)))  # [num_nodes, num_slots]
    actual_schedule_matrix = []
    for n in range(sel_num_nodes):
        actual_schedule_matrix.append([sel_schedule[s, n].item() for s in range(sel_num_slots)])

    # Counts for validity analysis
    act_flat = [act for row in actual_schedule_matrix for act in row]
    pred_flat = [p for row in pred_schedule_matrix for p in row]

    act_tag_cnt = act_flat.count(1)
    pred_tag_cnt = pred_flat.count(1)
    act_cg_cnt = act_flat.count(2)
    pred_cg_cnt = pred_flat.count(2)
    act_off_cnt = act_flat.count(0)
    pred_off_cnt = pred_flat.count(0)

    # 10. Required Final Output Report Format
    print("============================================================")
    print("GNN TRAINING COMPLETE")
    print("============================================================")
    print()
    print("Dataset:")
    print(f"500 graphs")
    print()
    print(f"Training graphs:")
    print(f"{len(train_graph_ids)}")
    print()
    print(f"Validation graphs:")
    print(f"{len(val_graph_ids)}")
    print()
    print(f"Test graphs:")
    print(f"{len(test_graph_ids)}")
    print()
    print(f"Training slot examples:")
    print(f"{len(train_examples)}")
    print()
    print(f"Validation slot examples:")
    print(f"{len(val_examples)}")
    print()
    print(f"Test slot examples:")
    print(f"{len(test_examples)}")
    print()
    print(f"Best Validation Accuracy:")
    print(f"{best_val_acc:.4f} (Epoch {best_epoch})")
    print()
    print(f"Best Validation Macro-F1:")
    print(f"{best_val_macro_f1:.4f} (Epoch {best_epoch})")
    print()
    print("============================================================")
    print("TEST RESULTS")
    print("============================================================")
    print()
    print(f"Test Accuracy:")
    print(f"{test_acc:.4f}")
    print()
    print(f"Test Macro-F1:")
    print(f"{test_macro_f1:.4f}")
    print()
    for i, cname in enumerate(CLASS_NAMES):
        print(f"{cname}:")
        print(f"Precision {precision[i]:.4f}")
        print(f"Recall {recall[i]:.4f}")
        print(f"F1 {f1[i]:.4f}")
        print()
    print("============================================================")
    print("CONFUSION MATRIX")
    print("============================================================")
    print()
    print("                 Predicted OFF  Predicted TAG   Predicted CG")
    print(f"Actual OFF (0):      {cm[0, 0]:>5}          {cm[0, 1]:>5}          {cm[0, 2]:>5}")
    print(f"Actual TAG (1):      {cm[1, 0]:>5}          {cm[1, 1]:>5}          {cm[1, 2]:>5}")
    print(f"Actual CG  (2):      {cm[2, 0]:>5}          {cm[2, 1]:>5}          {cm[2, 2]:>5}")
    print()
    print("============================================================")
    print("PREDICTION DISTRIBUTION")
    print("============================================================")
    print()
    print("Actual:")
    print(f"OFF {actual_counts[0]}")
    print(f"TAG {actual_counts[1]}")
    print(f"CG  {actual_counts[2]}")
    print()
    print("Predicted:")
    print(f"OFF {pred_counts[0]}")
    print(f"TAG {pred_counts[1]}")
    print(f"CG  {pred_counts[2]}")
    print()
    print(f"All three classes predicted: {all_three_predicted}")
    print()
    print("============================================================")
    print("TEST GRAPH EXAMPLE")
    print("============================================================")
    print()
    print(f"Graph ID: {sel_graph_id}")
    print(f"Number of nodes: {sel_num_nodes}")
    print(f"Number of tags: {sel_num_tags}")
    print(f"Number of schedule slots: {sel_num_slots}")
    print()
    print("ACTUAL SCHEDULE")
    header_slots = " ".join([f"Slot{s:>2}" for s in range(sel_num_slots)])
    print(f"          {header_slots}")
    for n in range(sel_num_nodes):
        row_str = " ".join([f"{CLASS_NAMES[actual_schedule_matrix[n][s]]:>6}" for s in range(sel_num_slots)])
        print(f"Node {n:<3} {row_str}")
    print()
    print("PREDICTED SCHEDULE")
    print(f"          {header_slots}")
    for n in range(sel_num_nodes):
        row_str = " ".join([f"{CLASS_NAMES[pred_schedule_matrix[n][s]]:>6}" for s in range(sel_num_slots)])
        print(f"Node {n:<3} {row_str}")
    print()
    print("Scheduling Validity Analysis:")
    print(f"Actual number of TAG actions:    {act_tag_cnt}")
    print(f"Predicted number of TAG actions: {pred_tag_cnt}")
    print(f"Actual number of CG actions:     {act_cg_cnt}")
    print(f"Predicted number of CG actions:  {pred_cg_cnt}")
    print(f"Actual number of OFF actions:    {act_off_cnt}")
    print(f"Predicted number of OFF actions: {pred_off_cnt}")
    print()
    tag_match = (pred_tag_cnt == sel_num_tags)
    print(f"Check: Predicted TAG count == actual number of hosted tags ({sel_num_tags})? {tag_match}")
    if not tag_match:
        diff = pred_tag_cnt - sel_num_tags
        print(f"Difference: {diff:+d} tags (Predicted {pred_tag_cnt} vs Hosted {sel_num_tags})")
    print()
    print("Note: Because the GCN predicts each slot independently,")
    print("predictions are NOT guaranteed to satisfy all OR-Tools scheduling constraints.")
    print()
    print("============================================================")
    print("MODEL FILE")
    print("============================================================")
    print()
    print("best_large_gcn_model.pt")
    print()
    print("Training history:")
    print("large_gnn_training_history.json")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train GCN on large 500-graph IoT scheduling dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH), help="Path to large_dataset.pt")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        dataset_path=args.dataset_path,
    )


if __name__ == "__main__":
    main()
