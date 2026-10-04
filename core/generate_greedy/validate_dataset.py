#!/usr/bin/env python3
"""
validate_dataset.py

Thorough validation script for dataset.pt and dataset_summary.json.
Validates:
- Structure and types of all graph samples
- Tensor shapes, dtypes, dimensions, and value ranges
- Absence of NaNs, Infs, empty tensors, or out-of-range node indices
- Correctness of target schedule labels (0=OFF, 1=TAG, 2=CG)
- Schedule tensor shape: (num_slots, num_nodes)
- Correlation between TAG actions and actual hosted tags
- Exact match of total TAG actions to 99 hosted tags
"""

import json
import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PT_FILE = SCRIPT_DIR / "dataset.pt"
SUMMARY_FILE = SCRIPT_DIR / "dataset_summary.json"


def validate():
    print("=" * 60)
    print("RGNN DATASET VALIDATION")
    print("=" * 60)
    print()

    # 1. Check existence
    if not PT_FILE.exists():
        print(f"ERROR: {PT_FILE} not found!")
        sys.exit(1)
    if not SUMMARY_FILE.exists():
        print(f"ERROR: {SUMMARY_FILE} not found!")
        sys.exit(1)

    # 2. Load files
    print(f"Loading {PT_FILE.name}...")
    dataset = torch.load(PT_FILE, weights_only=False)
    print(f"Loading {SUMMARY_FILE.name}...")
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    num_graphs = len(dataset)
    print(f"Successfully loaded {num_graphs} graphs.\n")

    # 3. Print structure of one complete graph sample (Sample 0)
    sample_0 = dataset[0]
    print("-" * 60)
    print("SAMPLE 0 COMPLETE STRUCTURE:")
    print(f"Graph ID: {sample_0['graph_id']} ({sample_0['file_name']})")
    print("-" * 60)
    for key, value in sample_0.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:<15}: Tensor shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, (list, dict)):
            val_str = str(value)
            if len(val_str) > 75:
                val_str = val_str[:72] + "..."
            print(f"  {key:<15}: {type(value).__name__} len={len(value)} -> {val_str}")
        else:
            print(f"  {key:<15}: {type(value).__name__} = {value}")
    print()

    # 4. Detailed Graph-by-Graph Verification
    print("-" * 60)
    print("GRAPH-BY-GRAPH VALIDATION:")
    print("-" * 60)

    total_nodes = 0
    total_tags = 0
    total_slots = 0
    total_tag_actions = 0
    global_action_counts = {0: 0, 1: 0, 2: 0}
    global_labels = set()

    for idx, sample in enumerate(dataset):
        gid = sample["graph_id"]
        fname = sample["file_name"]
        num_nodes = sample["num_nodes"]
        num_tags = sample["num_tags"]
        num_slots = sample["num_slots"]
        cg_threshold = sample["cg_threshold"]

        node_features = sample["node_features"]
        edge_index = sample["edge_index"]
        edge_weight = sample["edge_weight"]
        positions = sample["positions"]
        schedule = sample["schedule"]
        tags = sample["tags"]

        total_nodes += num_nodes
        total_tags += num_tags
        total_slots += num_slots

        # Check tensor shapes
        assert node_features.shape == (num_nodes, 4), f"[{gid}] node_features shape {node_features.shape} != ({num_nodes}, 4)"
        assert positions.shape == (num_nodes, 2), f"[{gid}] positions shape {positions.shape} != ({num_nodes}, 2)"
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, f"[{gid}] edge_index shape {edge_index.shape} invalid"
        num_edges = edge_index.shape[1]
        assert edge_weight.shape == (num_edges,), f"[{gid}] edge_weight shape {edge_weight.shape} != ({num_edges},)"
        assert schedule.shape == (num_slots, num_nodes), f"[{gid}] schedule shape {schedule.shape} != ({num_slots}, {num_nodes})"

        # Check types
        assert node_features.dtype == torch.float32, f"[{gid}] node_features dtype {node_features.dtype} != float32"
        assert positions.dtype == torch.float32, f"[{gid}] positions dtype {positions.dtype} != float32"
        assert edge_index.dtype == torch.int64, f"[{gid}] edge_index dtype {edge_index.dtype} != int64"
        assert edge_weight.dtype == torch.float32, f"[{gid}] edge_weight dtype {edge_weight.dtype} != float32"
        assert schedule.dtype == torch.int64, f"[{gid}] schedule dtype {schedule.dtype} != int64"

        # Check for NaN / Inf / Empty
        for t_name, tensor in [
            ("node_features", node_features),
            ("edge_index", edge_index),
            ("edge_weight", edge_weight),
            ("positions", positions),
            ("schedule", schedule),
        ]:
            assert tensor.numel() > 0, f"[{gid}] {t_name} is empty!"
            assert not torch.isnan(tensor).any(), f"[{gid}] {t_name} contains NaN!"
            assert not torch.isinf(tensor).any(), f"[{gid}] {t_name} contains Inf!"

        # Check edge index validity
        assert (edge_index >= 0).all() and (edge_index < num_nodes).all(), f"[{gid}] edge_index contains invalid node ID"

        # Check schedule labels
        unique_labels = set(schedule.unique().tolist())
        global_labels.update(unique_labels)
        assert unique_labels.issubset({0, 1, 2}), f"[{gid}] invalid schedule labels: {unique_labels}"

        # Action counts
        for lbl in (0, 1, 2):
            global_action_counts[lbl] += (schedule == lbl).sum().item()

        # Check tag actions vs hosted tags
        hosted_tags_count = {n: 0 for n in range(num_nodes)}
        for t in tags:
            assert 0 <= t["host"] < num_nodes, f"[{gid}] Tag host {t['host']} out of range"
            hosted_tags_count[t["host"]] += 1
        assert len(tags) == num_tags, f"[{gid}] tags length {len(tags)} != num_tags {num_tags}"

        for n in range(num_nodes):
            n_tag_actions = (schedule[:, n] == 1).sum().item()
            assert n_tag_actions == hosted_tags_count[n], (
                f"[{gid}] Node {n} tag actions ({n_tag_actions}) != hosted tags ({hosted_tags_count[n]})"
            )
            total_tag_actions += n_tag_actions

        print(
            f"  {fname:<16}: nodes={num_nodes:>2}, tags={num_tags:>2}, slots={num_slots:>2}, "
            f"edge_index={str(tuple(edge_index.shape)):>9}, schedule={str(tuple(schedule.shape)):>8}, "
            f"labels={sorted(unique_labels)} [OK]"
        )

    print()
    print("-" * 60)
    print("GLOBAL CHECKS & VERIFICATIONS:")
    print("-" * 60)

    # Global checks
    assert num_graphs == 13, f"Expected 13 graphs, found {num_graphs}"
    assert total_nodes == 100, f"Expected 100 total nodes, found {total_nodes}"
    assert total_tags == 99, f"Expected 99 total tags, found {total_tags}"
    assert total_slots == 29, f"Expected 29 total slots, found {total_slots}"
    assert total_tag_actions == 99, f"Expected 99 total TAG actions, found {total_tag_actions}"
    assert global_labels == {0, 1, 2}, f"Expected global labels {{0, 1, 2}}, found {global_labels}"

    print(f"  [PASS] Number of graphs processed        : {num_graphs} / 13")
    print(f"  [PASS] Total active nodes                : {total_nodes} / 100")
    print(f"  [PASS] Total tags hosted                 : {total_tags} / 99")
    print(f"  [PASS] Total schedule slots              : {total_slots} / 29")
    print(f"  [PASS] Total TAG actions across dataset  : {total_tag_actions} (matches expected 99 exactly)")
    print(f"  [PASS] Global schedule labels            : {sorted(global_labels)} (only 0=OFF, 1=TAG, 2=CG)")
    print(f"  [PASS] Schedule tensor shape convention  : (num_slots, num_nodes)")
    print(f"  [PASS] Node feature tensor shape         : (num_nodes, 4)")
    print(f"  [PASS] Position tensor shape             : (num_nodes, 2)")
    print(f"  [PASS] Edge index / weight shapes        : (2, 2*E) / (2*E,)")
    print(f"  [PASS] Node index range in edge_index    : all in [0, num_nodes - 1]")
    print(f"  [PASS] NaN / Inf / empty tensor check    : None found")
    print(f"  [PASS] Every TAG action hosts valid tag  : Verified for all nodes in all graphs")
    print()

    print("-" * 60)
    print("ACTION LABEL DISTRIBUTION:")
    print("-" * 60)
    print(f"  0 (OFF) : {global_action_counts[0]:>4}")
    print(f"  1 (TAG) : {global_action_counts[1]:>4}")
    print(f"  2 (CG)  : {global_action_counts[2]:>4}")
    print(f"  Total   : {sum(global_action_counts.values()):>4}")
    print()

    # Verify summary consistency
    assert summary["number_of_graphs"] == num_graphs
    assert summary["total_nodes"] == total_nodes
    assert summary["total_tags"] == total_tags
    assert summary["total_schedule_slots"] == total_slots
    assert summary["action_distribution"]["OFF"] == global_action_counts[0]
    assert summary["action_distribution"]["TAG"] == global_action_counts[1]
    assert summary["action_distribution"]["CG"] == global_action_counts[2]
    print("  [PASS] dataset_summary.json strictly matches dataset.pt")
    print()

    print("=" * 60)
    print("DATASET VALIDATION: PASSED")
    print("=" * 60)
    print()
    print("Conclusion: dataset.pt is completely verified and ready for GNN training.")


if __name__ == "__main__":
    validate()
