#!/usr/bin/env python3
"""
validate_large_dataset.py

Standalone validator for large_dataset.pt and large_dataset_summary.json.
Validates:
- All samples load properly
- Tensor dimensions and shapes
- Edge index validity within [0, num_nodes - 1]
- Absence of NaNs, Infs, or empty tensors
- Schedule labels strictly in {0, 1, 2}
- Invariant: TAG count in schedule == number of hosted tags for every graph
- Strict correspondence between large_dataset.pt and large_dataset_summary.json
"""

import json
import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PT_FILE = SCRIPT_DIR / "large_dataset.pt"
SUMMARY_FILE = SCRIPT_DIR / "large_dataset_summary.json"


def validate():
    print("=" * 60)
    print("LARGE DATASET VALIDATION")
    print("=" * 60)
    print()

    if not PT_FILE.exists():
        print(f"ERROR: {PT_FILE} does not exist!")
        sys.exit(1)
    if not SUMMARY_FILE.exists():
        print(f"ERROR: {SUMMARY_FILE} does not exist!")
        sys.exit(1)

    print(f"Loading {PT_FILE.name}...")
    dataset = torch.load(PT_FILE, weights_only=False)
    print(f"Loading {SUMMARY_FILE.name}...")
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    num_graphs = len(dataset)
    print(f"Loaded {num_graphs} graphs.\n")

    assert num_graphs == summary["total_successful"], (
        f"Mismatch: dataset has {num_graphs} graphs, summary reports {summary['total_successful']}"
    )

    total_tag_actions = 0
    all_labels = set()
    action_counts = {0: 0, 1: 0, 2: 0}

    for idx, sample in enumerate(dataset):
        gid = sample["graph_id"]
        num_nodes = sample["num_nodes"]
        num_tags = sample["num_tags"]
        num_slots = sample["num_slots"]

        node_features = sample["node_features"]
        edge_index = sample["edge_index"]
        edge_weight = sample["edge_weight"]
        positions = sample["positions"]
        schedule = sample["schedule"]
        tags = sample["tags"]

        # 1. Shapes
        assert node_features.shape == (num_nodes, 4), f"[{gid}] node_features shape mismatch"
        assert positions.shape == (num_nodes, 2), f"[{gid}] positions shape mismatch"
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, f"[{gid}] edge_index shape mismatch"
        assert edge_weight.shape == (edge_index.shape[1],), f"[{gid}] edge_weight shape mismatch"
        assert schedule.shape == (num_slots, num_nodes), f"[{gid}] schedule shape mismatch"

        # 2. No NaN / Inf / Empty
        for t_name, t in [
            ("node_features", node_features),
            ("edge_index", edge_index),
            ("edge_weight", edge_weight),
            ("positions", positions),
            ("schedule", schedule),
        ]:
            assert t.numel() > 0, f"[{gid}] {t_name} is empty!"
            assert not torch.isnan(t).any(), f"[{gid}] {t_name} contains NaN!"
            assert not torch.isinf(t).any(), f"[{gid}] {t_name} contains Inf!"

        # 3. Valid edge indices
        assert (edge_index >= 0).all() and (edge_index < num_nodes).all(), f"[{gid}] invalid edge index"

        # 4. Schedule labels in {0, 1, 2}
        labels = set(schedule.unique().tolist())
        all_labels.update(labels)
        assert labels.issubset({0, 1, 2}), f"[{gid}] invalid labels: {labels}"

        # 5. Invariant: TAG count in schedule == hosted tags count
        tag_actions = (schedule == 1).sum().item()
        assert tag_actions == num_tags, f"[{gid}] TAG actions ({tag_actions}) != num_tags ({num_tags})"
        total_tag_actions += tag_actions

        for lbl in (0, 1, 2):
            action_counts[lbl] += (schedule == lbl).sum().item()

        # 6. Verify each node with TAG action hosts tags
        hosted_tags_per_node = {n: 0 for n in range(num_nodes)}
        for t in tags:
            assert 0 <= t["host"] < num_nodes, f"[{gid}] tag host out of range"
            hosted_tags_per_node[t["host"]] += 1

        for n in range(num_nodes):
            n_tag_acts = (schedule[:, n] == 1).sum().item()
            assert n_tag_acts == hosted_tags_per_node[n], (
                f"[{gid}] node {n} tag actions ({n_tag_acts}) != hosted tags ({hosted_tags_per_node[n]})"
            )

    assert all_labels.issubset({0, 1, 2}), f"Unexpected labels in dataset: {all_labels}"
    assert total_tag_actions == summary["total_tags"], "Total TAG actions != total tags in summary"
    assert action_counts[0] == summary["action_distribution"]["OFF"], "OFF count mismatch"
    assert action_counts[1] == summary["action_distribution"]["TAG"], "TAG count mismatch"
    assert action_counts[2] == summary["action_distribution"]["CG"], "CG count mismatch"

    print("  [PASS] All samples loaded successfully")
    print(f"  [PASS] Verified {len(dataset)} graphs, {summary['total_nodes']} nodes, {summary['total_tags']} tags, {summary['total_slots']} slots")
    print("  [PASS] All tensor dimensions and types match specifications")
    print("  [PASS] All edge indices are within [0, num_nodes - 1]")
    print(f"  [PASS] Schedule labels strictly in {{0, 1, 2}} -> {sorted(all_labels)}")
    print("  [PASS] Zero NaNs, Infs, or empty tensors")
    print(f"  [PASS] Schedule invariant holds: {total_tag_actions} total TAG actions == {summary['total_tags']} total tags")
    print(f"  [PASS] Action distribution matches summary: OFF={action_counts[0]}, TAG={action_counts[1]}, CG={action_counts[2]}")
    print()
    print("============================================================")
    print("LARGE DATASET VALIDATION: PASSED")
    print("============================================================")


if __name__ == "__main__":
    validate()
