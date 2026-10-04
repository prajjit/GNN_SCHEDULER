#!/usr/bin/env python3
"""
train_gnn.py

Baseline Graph Convolutional Network (GCN) for IoT topology scheduling.
Predicts the scheduling action (0=OFF, 1=TAG, 2=CG) for each node at each schedule slot.
- Loads core/generate_greedy/dataset.pt
- Augments node features with normalized slot information: [degree, hosted_tags, x, y, slot_fraction]
- Splits by graph (70% train, 15% val, 15% test) with fixed SEED=42
- Computes class weights on the training set to address class imbalance
- Trains a 2-layer GCN with CrossEntropyLoss and early stopping on validation macro-F1
- Saves best_gcn_model.pt and training_history.json
- Evaluates on held-out test graphs with accuracy, macro-F1, per-class metrics, and confusion matrix
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch_geometric.nn import GCNConv

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "dataset.pt"
MODEL_SAVE_PATH = SCRIPT_DIR / "best_gcn_model.pt"
HISTORY_SAVE_PATH = SCRIPT_DIR / "training_history.json"

CLASS_NAMES = ["OFF", "TAG", "CG"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineGCN(nn.Module):
    """
    2-layer Graph Convolutional Network baseline.
    Input: [num_nodes, 5] (degree, hosted_tags, x, y, slot_fraction)
    Output: [num_nodes, 3] (logits for OFF, TAG, CG)
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
    Decompose each graph into slot-level training/eval examples.
    For each slot:
      slot_fraction = slot_index / max(1, num_slots - 1)
      node features = [degree, hosted_tags, x, y, slot_fraction]
      target = schedule[slot] (shape: [num_nodes])
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
                    "slot": s,
                    "num_nodes": num_nodes,
                    "x": x_with_slot,
                    "edge_index": edge_index,
                    "edge_weight": edge_weight,
                    "y": y,
                }
            )
    return examples


def compute_split_class_distribution(examples):
    counts = {0: 0, 1: 0, 2: 0}
    for ex in examples:
        for lbl in (0, 1, 2):
            counts[lbl] += (ex["y"] == lbl).sum().item()
    return counts


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
    # For 13 graphs: 9 train, 2 val, 2 test
    indices = list(range(num_graphs))
    random.shuffle(indices)

    train_graph_indices = indices[:9]
    val_graph_indices = indices[9:11]
    test_graph_indices = indices[11:]

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

    # 5. Print Initial GCN Baseline Info
    print("============================================================")
    print("GCN BASELINE")
    print("============================================================")
    print()
    print("Dataset:")
    print(f"Graphs: {num_graphs}")
    print(f"Nodes: {total_nodes}")
    print(f"Tags: {total_tags}")
    print(f"Slots: {total_slots}")
    print()
    print("Split:")
    print(f"Train graphs ({len(train_graph_ids)}): {train_graph_ids}")
    print(f"Validation graphs ({len(val_graph_ids)}): {val_graph_ids}")
    print(f"Test graphs ({len(test_graph_ids)}): {test_graph_ids}")
    print()
    print("Class distribution (Overall):")
    print(f"OFF: {dataset_actions[0]}")
    print(f"TAG: {dataset_actions[1]}")
    print(f"CG: {dataset_actions[2]}")
    print()
    print("Training set class counts & balanced weights:")
    print(f"OFF: count={int(train_class_counts[0].item())}, weight={class_weights[0].item():.4f}")
    print(f"TAG: count={int(train_class_counts[1].item())}, weight={class_weights[1].item():.4f}")
    print(f"CG : count={int(train_class_counts[2].item())}, weight={class_weights[2].item():.4f}")
    print()
    print("Model:")
    print("GCN layers: 2 (GCNConv 5->64, GCNConv 64->64, Linear 64->3)")
    print("Hidden dimension: 64")
    print("Output classes: 3")
    print()

    # 6. Initialize Model & Optimizer
    model = BaselineGCN(in_channels=5, hidden_channels=64, num_classes=3, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. Training Loop
    print("============================================================")
    print("TRAINING")
    print("============================================================")

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_preds = []
        train_targets_list = []

        # Shuffle training examples each epoch
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

        train_loss = epoch_loss / len(train_examples)
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

    # Save best model
    torch.save(best_model_state, MODEL_SAVE_PATH)
    print()
    print(f"Best model saved to: {MODEL_SAVE_PATH.name}")

    # 8. Test Evaluation
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_macro_f1, test_preds, test_targets = evaluate(
        model, test_examples, criterion
    )

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_preds, labels=[0, 1, 2], zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])

    # Prediction counts vs Actual counts
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
            "train_graphs": train_graph_ids,
            "val_graphs": val_graph_ids,
            "test_graphs": test_graph_ids,
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
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

    # 9. Print Test Results
    print("============================================================")
    print("TEST RESULTS")
    print("============================================================")
    print()
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print()
    print("Per-class results:")
    for i, cname in enumerate(CLASS_NAMES):
        print(
            f"  {cname:<4}: Precision = {precision[i]:.4f}, "
            f"Recall = {recall[i]:.4f}, "
            f"F1 = {f1[i]:.4f} (Support: {support[i]})"
        )
    print()
    print("Confusion Matrix:")
    print("                 Predicted OFF  Predicted TAG   Predicted CG")
    print(f"Actual OFF (0):      {cm[0, 0]:>5}          {cm[0, 1]:>5}          {cm[0, 2]:>5}")
    print(f"Actual TAG (1):      {cm[1, 0]:>5}          {cm[1, 1]:>5}          {cm[1, 2]:>5}")
    print(f"Actual CG  (2):      {cm[2, 0]:>5}          {cm[2, 1]:>5}          {cm[2, 2]:>5}")
    print()
    print("Prediction distribution:")
    for c in (0, 1, 2):
        cname = CLASS_NAMES[c]
        print(f"  {cname:<4}: Predicted = {pred_counts[c]:>2}, Actual = {actual_counts[c]:>2}")
    print()
    print(f"All three classes predicted: {all_three_predicted}")
    print()
    print("============================================================")
    print("IMPORTANT:")
    print("This is the baseline GCN.")
    print("The final RobustGANTT implementation will use a more advanced")
    print("Transformer-GNN/autoregressive architecture.")
    print("============================================================")
    print()
    print("Model Limitations Note:")
    print("- This baseline predicts scheduling actions for each time slot independently.")
    print("- It does NOT enforce global scheduling constraints such as:")
    print("    * Every tag must be scheduled exactly once across the schedule")
    print("    * Carrier/interrogation RF compatibility in the same slot")
    print("    * Sequential tag constraints on co-hosted tags")
    print("    * Minimum carrier signal strength threshold (cg_threshold)")
    print("    * Globally optimal schedule objective (e.g. minimal slot count)")
    print("- Those constraints remain guaranteed by the OR-Tools optimizer labels.")
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="Train GCN baseline on IoT graph scheduling dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH), help="Path to dataset.pt")
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
