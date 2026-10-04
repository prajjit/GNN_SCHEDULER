#!/usr/bin/env python3
"""
generate_large_dataset.py

Generates a large-scale (approx 500 graphs) diverse synthetic IoT topology
and schedule dataset for RGNN using existing topology generators in topology.py
and the OR-Tools scheduler in schedule.py.
- Uses ErdosTopologyGraph, GridTopologyGraph, GeometricTopologyGraph, and RealisticGeometricTopologyGraph
- Varies node counts, tag counts, edge densities, positions, and carrier thresholds
- Solves schedules using compute_schedule_ortools() with safe greedy fallback on 64-bit integer overflow
- Strictly preserves target labels (0=OFF, 1=TAG, 2=CG)
- Validates schedule integrity (TAG count in schedule == number of hosted tags)
- Saves large_dataset.pt and large_dataset_summary.json
- Automatically runs complete dataset validation
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from schedule import compute_schedule_ortools
from topology import (
    GeometricTopologyGraph,
    GridTopologyGraph,
    RealisticGeometricTopologyGraph,
    TopologyGraph,
    gen_connected_with_retry,
)

OUTPUT_PT = SCRIPT_DIR / "large_dataset.pt"
OUTPUT_SUMMARY = SCRIPT_DIR / "large_dataset_summary.json"

TOPOLOGY_TYPES = ["geometric", "lattice", "random", "geometricr"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_topology(topo_type, n_nodes, n_tags):
    """
    Generate a connected TopologyGraph using project's existing topology classes/methods.
    Assigns realistic active-node wireless edge strengths where applicable.
    """
    if topo_type == "geometric":
        radius = random.uniform(0.45, 0.85)
        topo = GeometricTopologyGraph(n_nodes=n_nodes, n_tags=n_tags, radius=radius)
        cg_thresh = random.choice([0, 20, 25, 30])
        for u, v in topo.edges():
            if u in topo.active_nodes and v in topo.active_nodes:
                topo.edges[u, v]["weight"] = random.randint(30, 65)

    elif topo_type == "lattice":
        # Grid side chosen so that grid_side * other_side >= n_nodes
        grid_side = random.choice([2, 3, 4])
        topo = GridTopologyGraph(n_nodes=n_nodes, n_tags=n_tags, grid_side=grid_side)
        cg_thresh = random.choice([0, 20, 25, 30])
        for u, v in topo.edges():
            if u in topo.active_nodes and v in topo.active_nodes:
                topo.edges[u, v]["weight"] = random.randint(30, 65)

    elif topo_type == "random":
        # Erdos-Renyi binomial graph with spring positions assigned
        p = random.uniform(0.40, 0.75)
        G = gen_connected_with_retry(nx.gnp_random_graph, n=n_nodes, p=p, max_attempts=50)
        pos = nx.spring_layout(G)
        for nid, (x, y) in pos.items():
            G.nodes[nid]["pos"] = (float(x), float(y))
        G.graph["topology"] = "random"
        G.graph["p"] = p
        topo = TopologyGraph(incoming_graph_data=G, n_nodes=n_nodes, n_tags=n_tags)
        cg_thresh = random.choice([0, 20, 25, 30])
        for u, v in topo.edges():
            if u in topo.active_nodes and v in topo.active_nodes:
                topo.edges[u, v]["weight"] = random.randint(30, 65)

    elif topo_type == "geometricr":
        side = random.choice([70, 90, 110, 130])
        topo = RealisticGeometricTopologyGraph(n_nodes=n_nodes, n_tags=n_tags, side=side)
        cg_thresh = 0  # RealisticGeometric uses path loss dB edge weights

    else:
        raise ValueError(f"Unknown topology type: {topo_type}")

    return topo, cg_thresh


def solve_schedule(topo, cg_threshold, timeout_ms=5000):
    """
    Attempt normal OR-Tools solve first; fall back to greedy=True on 64-bit int overflow.
    """
    try:
        sched = compute_schedule_ortools(
            topo,
            cg_threshold=cg_threshold,
            timeout=timeout_ms,
            optimize=True,
            workers=0,
            greedy=False,
        )
        mode = "normal"
    except (TypeError, OverflowError):
        # 64-bit integer overflow fallback for high tag count
        sched = compute_schedule_ortools(
            topo,
            cg_threshold=cg_threshold,
            timeout=timeout_ms,
            optimize=True,
            workers=0,
            greedy=True,
        )
        mode = "greedy_fallback"

    return sched, mode


def extract_sample_dict(graph_id, topo, sched, topo_type, cg_threshold, solver_mode):
    """
    Extract tensors and metadata from a valid TopologyGraph and Schedule object.
    Matches the schema of dataset.pt.
    """
    num_nodes = len(topo.active_nodes)
    num_tags = len(topo.tags)
    num_slots = len(sched)

    # Sort active node labels by integer ID (A0, A1, ...)
    sorted_node_labels = sorted(topo.active_nodes, key=TopologyGraph.node_id_from_label)

    # 1. Positions
    positions_list = []
    for nl in sorted_node_labels:
        pos = topo.nodes[nl].get("pos", (0.0, 0.0))
        positions_list.append([float(pos[0]), float(pos[1])])
    positions_tensor = torch.tensor(positions_list, dtype=torch.float32)

    # 2. Node features: [degree, hosted_tags, x, y]
    node_features_list = []
    for idx, nl in enumerate(sorted_node_labels):
        deg = float(topo.active_degree(nl))
        hosted_tags_cnt = float(len(topo.get_hosted_tags(nl)))
        x, y = positions_list[idx][0], positions_list[idx][1]
        node_features_list.append([deg, hosted_tags_cnt, x, y])
    node_features_tensor = torch.tensor(node_features_list, dtype=torch.float32)

    # 3. Connectivity (bidirectional edges between active nodes)
    edges = []
    weights = []
    for u_lbl, v_lbl, data in topo.edges(data=True):
        if u_lbl in topo.active_nodes and v_lbl in topo.active_nodes:
            u = TopologyGraph.node_id_from_label(u_lbl)
            v = TopologyGraph.node_id_from_label(v_lbl)
            w = float(data.get("weight", 1.0))
            edges.append((u, v))
            weights.append(w)
            edges.append((v, u))
            weights.append(w)

    if edges:
        sorted_pairs = sorted(zip(edges, weights), key=lambda item: (item[0][0], item[0][1]))
        src = [p[0][0] for p in sorted_pairs]
        dst = [p[0][1] for p in sorted_pairs]
        w_sorted = [p[1] for p in sorted_pairs]
        edge_index_tensor = torch.tensor([src, dst], dtype=torch.long)
        edge_weight_tensor = torch.tensor(w_sorted, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float32)

    # 4. Tags metadata
    tags_list = []
    for t_label in sorted(topo.tags, key=TopologyGraph.tag_id_from_label):
        tag_id = TopologyGraph.tag_id_from_label(t_label)
        host_label = topo.nodes[t_label]["host"]
        host_id = TopologyGraph.node_id_from_label(host_label)
        tags_list.append({"id": tag_id, "host": host_id})

    # 5. Schedule target tensor: shape (num_slots, num_nodes)
    sched_matrix = []
    action_counts = {"OFF": 0, "TAG": 0, "CG": 0}
    for s in range(num_slots):
        slot_actions = []
        for nl in sorted_node_labels:
            act = sched[nl][s]
            if act == "OFF":
                slot_actions.append(0)
                action_counts["OFF"] += 1
            elif act == "CG":
                slot_actions.append(2)
                action_counts["CG"] += 1
            else:
                # Any tag string (e.g. 'T0', 'T1', etc.)
                slot_actions.append(1)
                action_counts["TAG"] += 1
        sched_matrix.append(slot_actions)

    schedule_tensor = torch.tensor(sched_matrix, dtype=torch.long)

    sample = {
        "graph_id": graph_id,
        "generation_type": topo_type,
        "solver_mode": solver_mode,
        "node_features": node_features_tensor,
        "edge_index": edge_index_tensor,
        "edge_weight": edge_weight_tensor,
        "positions": positions_tensor,
        "tags": tags_list,
        "schedule": schedule_tensor,
        "num_nodes": num_nodes,
        "num_tags": num_tags,
        "num_slots": num_slots,
        "cg_threshold": cg_threshold,
    }

    stats = {
        "graph_id": graph_id,
        "generation_type": topo_type,
        "solver_mode": solver_mode,
        "num_nodes": num_nodes,
        "num_tags": num_tags,
        "num_edges": len(edges) // 2,
        "num_slots": num_slots,
        "cg_threshold": cg_threshold,
        "action_distribution": action_counts,
    }

    return sample, stats


def generate_large_dataset(target_samples=500, max_attempts=700, seed=42):
    set_seed(seed)

    print("============================================================")
    print(f"LARGE DATASET GENERATION (Target: {target_samples} graphs)")
    print("============================================================")
    print()

    dataset = []
    metadata_list = []
    failure_reasons = Counter()

    total_requested = target_samples
    total_successful = 0
    total_failed = 0

    topo_type_counts = Counter()
    solver_mode_counts = Counter()
    global_action_counts = {"OFF": 0, "TAG": 0, "CG": 0}

    attempt = 0
    start_time = time.time()

    while len(dataset) < target_samples and attempt < max_attempts:
        attempt += 1
        ttype = TOPOLOGY_TYPES[attempt % len(TOPOLOGY_TYPES)]

        # Sample realistic node count (4 to 12) and tag count (2 to 14)
        n_nodes = random.randint(4, 12)
        max_t = min(14, max(2, int(n_nodes * 1.8)))
        n_tags = random.randint(2, max_t)

        try:
            topo, cg_thresh = create_topology(ttype, n_nodes, n_tags)
        except Exception as e:
            total_failed += 1
            failure_reasons[f"Topology creation failed: {type(e).__name__}"] += 1
            continue

        # Solve schedule
        try:
            sched, solver_mode = solve_schedule(topo, cg_thresh, timeout_ms=5000)
        except Exception as e:
            total_failed += 1
            failure_reasons[f"Scheduler exception: {type(e).__name__}"] += 1
            continue

        if sched is None or len(sched) == 0:
            total_failed += 1
            failure_reasons["Scheduler returned None (UNSATISFIABLE)"] += 1
            continue

        # Extract features and format
        graph_id = f"synth_{len(dataset) + 1:04d}"
        try:
            sample, stats = extract_sample_dict(
                graph_id, topo, sched, ttype, cg_thresh, solver_mode
            )
        except Exception as e:
            total_failed += 1
            failure_reasons[f"Feature extraction error: {type(e).__name__}"] += 1
            continue

        # Strict quality checks
        schedule_tensor = sample["schedule"]
        num_nodes_actual = sample["num_nodes"]
        num_tags_actual = sample["num_tags"]
        num_slots_actual = sample["num_slots"]

        if num_nodes_actual == 0 or num_tags_actual == 0 or num_slots_actual == 0:
            total_failed += 1
            failure_reasons["Zero nodes, tags, or slots"] += 1
            continue

        if schedule_tensor.shape != (num_slots_actual, num_nodes_actual):
            total_failed += 1
            failure_reasons["Schedule shape mismatch"] += 1
            continue

        # Invariant: TAG count in schedule must equal number of hosted tags
        tag_actions_in_sched = (schedule_tensor == 1).sum().item()
        if tag_actions_in_sched != num_tags_actual:
            total_failed += 1
            failure_reasons["TAG action count != hosted tags invariant violated"] += 1
            continue

        # Accepted sample
        dataset.append(sample)
        metadata_list.append(stats)
        total_successful += 1

        topo_type_counts[ttype] += 1
        solver_mode_counts[solver_mode] += 1
        for act, cnt in stats["action_distribution"].items():
            global_action_counts[act] += cnt

        # Progress reporting every 25 successful graphs
        if total_successful % 25 == 0:
            avg_nodes = sum(s["num_nodes"] for s in metadata_list) / total_successful
            avg_tags = sum(s["num_tags"] for s in metadata_list) / total_successful
            print(
                f"Generated: {total_successful} / {target_samples} | "
                f"Successful: {total_successful} | "
                f"Failed: {total_failed} | "
                f"Average nodes: {avg_nodes:.1f} | "
                f"Average tags: {avg_tags:.1f}"
            )

    elapsed = time.time() - start_time
    print()

    # Save large_dataset.pt
    OUTPUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUTPUT_PT)

    # Compute summary stats
    all_nodes = [s["num_nodes"] for s in metadata_list]
    all_tags = [s["num_tags"] for s in metadata_list]
    all_slots = [s["num_slots"] for s in metadata_list]

    summary_data = {
        "total_requested": total_requested,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "failure_reasons": dict(failure_reasons),
        "total_nodes": sum(all_nodes),
        "total_tags": sum(all_tags),
        "total_slots": sum(all_slots),
        "action_distribution": global_action_counts,
        "topology_type_distribution": dict(topo_type_counts),
        "solver_mode_distribution": dict(solver_mode_counts),
        "min_nodes": min(all_nodes) if all_nodes else 0,
        "max_nodes": max(all_nodes) if all_nodes else 0,
        "min_tags": min(all_tags) if all_tags else 0,
        "max_tags": max(all_tags) if all_tags else 0,
        "min_slots": min(all_slots) if all_slots else 0,
        "max_slots": max(all_slots) if all_slots else 0,
        "elapsed_seconds": round(elapsed, 2),
        "graph_by_graph_metadata": metadata_list,
    }

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Print final summary matching requirement 9
    print("============================================================")
    print("LARGE DATASET GENERATION COMPLETE")
    print("============================================================")
    print()
    print(f"Requested: {total_requested}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print()
    print(f"Total nodes: {summary_data['total_nodes']}")
    print(f"Total tags: {summary_data['total_tags']}")
    print(f"Total slots: {summary_data['total_slots']}")
    print()
    print(f"OFF: {global_action_counts['OFF']}")
    print(f"TAG: {global_action_counts['TAG']}")
    print(f"CG: {global_action_counts['CG']}")
    print()
    print(f"Normal OR-Tools: {solver_mode_counts['normal']}")
    print(f"Greedy fallback: {solver_mode_counts['greedy_fallback']}")
    print()
    print("Dataset:")
    print(f"{OUTPUT_PT.name}")
    print()
    print("Summary:")
    print(f"{OUTPUT_SUMMARY.name}")
    print()

    return dataset, summary_data


def validate_large_dataset(pt_path=OUTPUT_PT, summary_path=OUTPUT_SUMMARY):
    """
    Thoroughly validate large_dataset.pt against quality and invariant criteria.
    """
    print("-" * 60)
    print("VALIDATING LARGE DATASET...")
    print("-" * 60)

    dataset = torch.load(pt_path, weights_only=False)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(dataset) == summary["total_successful"], "Dataset count mismatch with summary"

    total_tag_actions = 0
    all_labels = set()

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

    print("  [PASS] All samples loaded successfully")
    print(f"  [PASS] Verified {len(dataset)} graphs, {summary['total_nodes']} nodes, {summary['total_tags']} tags")
    print("  [PASS] All tensor dimensions and types match specifications")
    print("  [PASS] All edge indices are within [0, num_nodes - 1]")
    print(f"  [PASS] Schedule labels strictly in {{0, 1, 2}} -> {sorted(all_labels)}")
    print("  [PASS] Zero NaNs, Infs, or empty tensors")
    print(f"  [PASS] Schedule invariant holds: {total_tag_actions} total TAG actions == {summary['total_tags']} total tags")
    print()
    print("============================================================")
    print("LARGE DATASET VALIDATION: PASSED")
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="Generate large diverse IoT scheduling dataset.")
    parser.add_argument("--samples", type=int, default=500, help="Number of successful graph samples (default: 500)")
    parser.add_argument("--max-attempts", type=int, default=700, help="Maximum attempts (default: 700)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_large_dataset(
        target_samples=args.samples,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    validate_large_dataset()


if __name__ == "__main__":
    main()
