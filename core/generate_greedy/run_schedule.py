#!/usr/bin/env python3
"""
run_schedule.py

Reads a JSON topology description, deserializes it into a `TopologyGraph`
(networkx-based), runs the CP-SAT (`compute_schedule_ortools`) scheduler,
and prints the resulting schedule.

Expected JSON schema:

{
  "nodes": [ {"id": 0, "pos": [0, 0]}, ... ],
  "edges": [ {"from": 0, "to": 1, "weight": 45}, ... ],
  "tags":  [ {"host": 1}, {"host": 2, "id": 3}, ... ],   # "id" optional
  "cg_threshold": 30,
  "solver": {
    "timeout_ms": 30000,
    "optimize": true,
    "workers": 0,
    "greedy": false
  }
}

Usage:
    python run_schedule.py topology.json
    python run_schedule.py topology.json --repo-path /path/to/scripts

Note: this script imports `topology.py` and `schedule.py` from the
original repo's `scripts/` directory. Point --repo-path at that directory
(or run this script from inside it / add it to PYTHONPATH) so the imports
resolve.
"""

import argparse
import json
import sys
from pathlib import Path


def load_repo_modules(repo_path):
    """Add the repo's scripts/ dir to sys.path so topology.py / schedule.py
    can be imported, then import and return them."""
    if repo_path is not None:
        sys.path.insert(0, str(Path(repo_path).resolve()))
    try:
        import networkx as nx
        from topology import TopologyGraph
        from schedule import compute_schedule_ortools
    except ImportError as e:
        raise SystemExit(
            f"Could not import required modules ({e}).\n"
            f"Pass --repo-path pointing at the directory containing "
            f"topology.py and schedule.py, or run this script from there."
        )
    return nx, TopologyGraph, compute_schedule_ortools


def build_topology(data, nx, TopologyGraph):
    """Deserialize the JSON dict into a TopologyGraph instance."""

    # --- basic validation -------------------------------------------------
    if "nodes" not in data or "edges" not in data:
        raise ValueError("JSON must contain 'nodes' and 'edges' keys.")

    node_ids = [n["id"] for n in data["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node ids found in 'nodes'.")

    # --- build the plain networkx graph -----------------------------------
    G = nx.Graph()
    for n in data["nodes"]:
        pos = tuple(n["pos"]) if "pos" in n else (0.0, 0.0)
        G.add_node(n["id"], pos=pos)

    valid_ids = set(node_ids)
    for e in data["edges"]:
        f, t = e["from"], e["to"]
        if f not in valid_ids or t not in valid_ids:
            raise ValueError(f"Edge ({f}, {t}) references an undefined node id.")
        weight = e.get("weight", 1)
        G.add_edge(f, t, weight=weight)

    # --- wrap as TopologyGraph, no auto-random tags -----------------------
    topo = TopologyGraph(incoming_graph_data=G, n_tags=0)

    # --- explicitly place tags ---------------------------------------------
    # Accept either {"host": 1} (auto-numbered Tn) or {"id": 3, "host": 1}
    tag_defs = data.get("tags", [])
    next_tag_id = 0
    tag_assignment = {}
    for t in tag_defs:
        host_id = t["host"]
        host_label = TopologyGraph.node_label_format(host_id)
        if host_label not in topo.nodes:
            raise ValueError(f"Tag host node {host_id} not found in topology.")
        tag_id = t.get("id", next_tag_id)
        next_tag_id = max(next_tag_id, tag_id + 1)
        tag_label = TopologyGraph.tag_label_format(tag_id)
        tag_assignment[tag_label] = host_label

    if tag_assignment:
        topo.add_tags(tag_assignment)

    return topo


def print_schedule(schedule, topology_data):
    """Pretty-print a Schedule object as a per-node, per-slot table."""
    if schedule is None:
        print("UNSATISFIABLE: no valid schedule exists for this topology "
              "at the given cg_threshold.")
        return

    n_slots = len(schedule)
    nodes = sorted(schedule.active_nodes,
                   key=lambda n: int(n[1:]))  # sort A0, A1, A2... numerically

    print(f"Schedule ({n_slots} slot{'s' if n_slots != 1 else ''}, "
          f"{schedule.carrier_slots} carrier activation"
          f"{'s' if schedule.carrier_slots != 1 else ''}):\n")

    header = "Node".ljust(6) + "".join(f"Slot{i}".rjust(8) for i in range(n_slots))
    print(header)
    print("-" * len(header))
    for node in nodes:
        row = node.ljust(6) + "".join(str(v).rjust(8) for v in schedule[node])
        print(row)

    print()
    obj = schedule.get_mnz_objectives()
    print(f"Objective breakdown:")
    print(f"  carrier slots used : {obj['num_cg_slots']}")
    print(f"  sum_cg_nodes       : {obj['sum_cg_nodes']}")
    print(f"  sum_int_slots      : {obj['sum_int_slots']}")
    print(f"  total (lexico obj) : {obj['total']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", help="Path to the topology JSON file")
    parser.add_argument("--repo-path", default=None,
                         help="Directory containing topology.py / schedule.py "
                              "(defaults to current PYTHONPATH)")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    nx, TopologyGraph, compute_schedule_ortools = load_repo_modules(args.repo_path)

    topo = build_topology(data, nx, TopologyGraph)

    cg_threshold = data.get("cg_threshold", 0)
    solver_opts = data.get("solver", {})
    timeout_ms = solver_opts.get("timeout_ms", 30000)
    optimize = solver_opts.get("optimize", True)
    workers = solver_opts.get("workers", 0)
    greedy = solver_opts.get("greedy", False)

    print(f"Loaded topology: {len(topo.active_nodes)} active nodes, "
          f"{len(topo.tags)} tags, cg_threshold={cg_threshold}\n")

    schedule = compute_schedule_ortools(
        topo,
        cg_threshold=cg_threshold,
        timeout=timeout_ms,
        optimize=optimize,
        workers=workers,
        greedy=greedy,
    )

    print_schedule(schedule, data)


if __name__ == "__main__":
    main()
