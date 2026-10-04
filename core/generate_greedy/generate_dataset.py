#!/usr/bin/env python3
"""
generate_dataset.py

Generates a machine-learning-ready graph scheduling dataset for RGNN.
Loads network topology JSON graphs from the graphs/ directory, invokes the
existing OR-Tools CP-SAT scheduler (compute_schedule_ortools) to obtain
optimal/greedy schedules, encodes scheduling actions into GNN target labels
(OFF=0, TAG=1, CG=2), and saves the resulting dataset as PyTorch tensors
along with a metadata summary JSON.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import networkx as nx
import torch

# Ensure local imports resolve when running from script directory or elsewhere
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_schedule import build_topology
from schedule import compute_schedule_ortools
from topology import TopologyGraph

# Target action label constants
LABEL_OFF = 0
LABEL_TAG = 1
LABEL_CG = 2


def natural_sort_key(path):
    """Sort filenames in natural order (e.g., graph-1, graph-2, ..., graph-10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


def action_to_label(action_str):
    """
    Convert a schedule action string to a GNN target label.
    - 'OFF' -> 0
    - Any tag (e.g. 'T0', 'T1', etc.) or 'TX' -> 1
    - 'CG' -> 2
    """
    if action_str == "OFF":
        return LABEL_OFF
    elif action_str == "CG":
        return LABEL_CG
    else:
        # Any tag identifier (T0, T1, ...) or interrogation action
        return LABEL_TAG


def solve_schedule(topo, data):
    """
    Compute schedule using the existing OR-Tools scheduler API.
    Handles solver options from JSON and safely falls back to greedy=True
    if integer domain overflow occurs in OR-Tools for topologies with large tag counts.
    """
    cg_threshold = data.get("cg_threshold", 0)
    solver_opts = data.get("solver", {})
    timeout_ms = solver_opts.get("timeout_ms", 30000)
    optimize = solver_opts.get("optimize", True)
    workers = solver_opts.get("workers", 0)
    greedy = solver_opts.get("greedy", False)

    try:
        schedule = compute_schedule_ortools(
            topo,
            cg_threshold=cg_threshold,
            timeout=timeout_ms,
            optimize=optimize,
            workers=workers,
            greedy=greedy,
        )
    except (TypeError, OverflowError):
        # When n_tags >= 15 with greedy=False, objective domain calculation in
        # OR-Tools overflows 64-bit integer. Fall back to greedy=True.
        schedule = compute_schedule_ortools(
            topo,
            cg_threshold=cg_threshold,
            timeout=timeout_ms,
            optimize=optimize,
            workers=workers,
            greedy=True,
        )

    if schedule is None:
        raise ValueError("Scheduler returned None (UNSATISFIABLE) for topology.")

    return schedule


def process_graph(file_path):
    """Process a single JSON graph file and produce a graph sample dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Build TopologyGraph exactly as run_schedule.py does
    topo = build_topology(data, nx, TopologyGraph)

    # 2. Compute schedule using existing OR-Tools scheduler
    schedule_obj = solve_schedule(topo, data)

    # 3. Extract nodes and positions
    num_nodes = len(data["nodes"])
    # Sort nodes by ID (0..num_nodes-1) to ensure deterministic order
    sorted_nodes = sorted(data["nodes"], key=lambda n: n["id"])

    # Positions: [x, y]
    positions_list = []
    for n in sorted_nodes:
        pos = n.get("pos", [0.0, 0.0])
        positions_list.append([float(pos[0]), float(pos[1])])
    positions_tensor = torch.tensor(positions_list, dtype=torch.float32)

    # 4. Extract node features: [degree, number_of_tags, x_position, y_position]
    node_features_list = []
    for n in sorted_nodes:
        nid = n["id"]
        node_label = TopologyGraph.node_label_format(nid)
        degree = float(topo.active_degree(node_label))
        n_hosted_tags = float(len(topo.get_hosted_tags(node_label)))
        pos_x = float(positions_list[nid][0])
        pos_y = float(positions_list[nid][1])
        node_features_list.append([degree, n_hosted_tags, pos_x, pos_y])
    node_features_tensor = torch.tensor(node_features_list, dtype=torch.float32)

    # 5. Extract connectivity: edge_index, edge_weight
    # Undirected graph representation (bidirectional edges)
    edges = []
    weights = []
    for e in data.get("edges", []):
        u, v = int(e["from"]), int(e["to"])
        w = float(e.get("weight", 1.0))
        edges.append((u, v))
        weights.append(w)
        edges.append((v, u))
        weights.append(w)

    if edges:
        # Sort by (source, destination)
        sorted_edge_data = sorted(zip(edges, weights), key=lambda x: (x[0][0], x[0][1]))
        src = [item[0][0] for item in sorted_edge_data]
        dst = [item[0][1] for item in sorted_edge_data]
        w_sorted = [item[1] for item in sorted_edge_data]
        edge_index_tensor = torch.tensor([src, dst], dtype=torch.long)
        edge_weight_tensor = torch.tensor(w_sorted, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float32)

    # 6. Extract tags
    tag_defs = data.get("tags", [])
    next_tag_id = 0
    tags_list = []
    for t in tag_defs:
        host_id = int(t["host"])
        tag_id = int(t.get("id", next_tag_id))
        next_tag_id = max(next_tag_id, tag_id + 1)
        tags_list.append({"id": tag_id, "host": host_id})
    num_tags = len(tags_list)

    # 7. Extract schedule labels: shape (num_slots, num_nodes)
    num_slots = len(schedule_obj)
    if num_slots == 0:
        raise ValueError(f"Schedule for {file_path.name} unexpectedly has 0 slots.")

    schedule_matrix = []
    graph_action_counts = {"OFF": 0, "TAG": 0, "CG": 0}

    for s in range(num_slots):
        slot_actions = []
        for n in sorted_nodes:
            nid = n["id"]
            node_label = TopologyGraph.node_label_format(nid)
            action_str = schedule_obj[node_label][s]
            label = action_to_label(action_str)
            slot_actions.append(label)
            if label == LABEL_OFF:
                graph_action_counts["OFF"] += 1
            elif label == LABEL_TAG:
                graph_action_counts["TAG"] += 1
            elif label == LABEL_CG:
                graph_action_counts["CG"] += 1
        schedule_matrix.append(slot_actions)

    schedule_tensor = torch.tensor(schedule_matrix, dtype=torch.long)
    cg_threshold = data.get("cg_threshold", 0)

    # Raw string schedule for reference
    raw_schedule = {
        TopologyGraph.node_label_format(n["id"]): schedule_obj[TopologyGraph.node_label_format(n["id"])]
        for n in sorted_nodes
    }

    sample = {
        "graph_id": file_path.stem,
        "file_name": file_path.name,
        "node_features": node_features_tensor,
        "edge_index": edge_index_tensor,
        "edge_weight": edge_weight_tensor,
        "positions": positions_tensor,
        "tags": tags_list,
        "num_nodes": num_nodes,
        "num_tags": num_tags,
        "num_slots": num_slots,
        "schedule": schedule_tensor,
        "cg_threshold": cg_threshold,
        "raw_schedule": raw_schedule,
    }

    stats = {
        "graph_id": file_path.stem,
        "file_name": file_path.name,
        "num_nodes": num_nodes,
        "num_tags": num_tags,
        "num_edges": len(data.get("edges", [])),
        "num_slots": num_slots,
        "cg_threshold": cg_threshold,
        "action_distribution": graph_action_counts,
    }

    return sample, stats


def generate_dataset(graphs_dir, output_pt, output_summary):
    """
    Load all JSON graph files from graphs_dir, generate schedules and features,
    and save dataset.pt and dataset_summary.json.
    """
    graphs_dir = Path(graphs_dir)
    output_pt = Path(output_pt)
    output_summary = Path(output_summary)

    json_files = sorted(graphs_dir.glob("*.json"), key=natural_sort_key)
    if not json_files:
        raise FileNotFoundError(f"No JSON graph files found in {graphs_dir}")

    print("========================================")
    print("DATASET GENERATION")
    print("========================================")
    print()

    dataset = []
    graph_stats = []
    total_nodes = 0
    total_tags = 0
    total_slots = 0
    action_counts = {"OFF": 0, "TAG": 0, "CG": 0}

    for fpath in json_files:
        print(f"Processing {fpath.name}")
        sample, stats = process_graph(fpath)

        dataset.append(sample)
        graph_stats.append(stats)

        total_nodes += stats["num_nodes"]
        total_tags += stats["num_tags"]
        total_slots += stats["num_slots"]

        for action, count in stats["action_distribution"].items():
            action_counts[action] += count

        print(f"Nodes: {stats['num_nodes']}")
        print(f"Tags: {stats['num_tags']}")
        print(f"Schedule slots: {stats['num_slots']}")
        print("Schedule generated successfully")
        print()

    # Save PyTorch dataset
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_pt)

    # Save summary JSON
    summary_data = {
        "number_of_graphs": len(dataset),
        "total_nodes": total_nodes,
        "total_tags": total_tags,
        "total_schedule_slots": total_slots,
        "action_distribution": action_counts,
        "graph-by-graph statistics": graph_stats,
        "graph_by_graph_statistics": graph_stats,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("========================================")
    print("DATASET COMPLETE")
    print("========================================")
    print()
    print(f"Graphs processed: {len(dataset)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total tags: {total_tags}")
    print(f"Total schedule slots: {total_slots}")
    print()
    print("Action distribution:")
    print(f"OFF: {action_counts['OFF']}")
    print(f"TAG: {action_counts['TAG']}")
    print(f"CG: {action_counts['CG']}")
    print()
    print(f"Dataset saved to:\n{output_pt.name}")
    print()
    print(f"Summary saved to:\n{output_summary.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate RGNN training dataset from topology graphs.")
    parser.add_argument(
        "--graphs-dir",
        default=str(SCRIPT_DIR / "graphs"),
        help="Directory containing topology JSON graph files (default: core/generate_greedy/graphs)",
    )
    parser.add_argument(
        "--output-pt",
        default=str(SCRIPT_DIR / "dataset.pt"),
        help="Path for generated PyTorch dataset file (default: core/generate_greedy/dataset.pt)",
    )
    parser.add_argument(
        "--output-summary",
        default=str(SCRIPT_DIR / "dataset_summary.json"),
        help="Path for dataset summary JSON file (default: core/generate_greedy/dataset_summary.json)",
    )
    args = parser.parse_args()

    generate_dataset(
        graphs_dir=args.graphs_dir,
        output_pt=args.output_pt,
        output_summary=args.output_summary,
    )


if __name__ == "__main__":
    main()
