<<<<<<< HEAD

# RGNN

scheduling for IoT networks with backscatter communication.

## Installation

Clone the repository:

```bash
git clone https://github.com/sugvn/rgnn.git
cd rgnn/core/generate_greedy
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Running

Run the scheduler with a graph input file:

```bash
python run_schedule.py graphs/graph-<n>.json
```

Replace `<n>` with the graph number you want to run.

For example:

```bash
python run_schedule.py graphs/graph-1.json
```
=======
# IoT Network Scheduling Using Graph Convolutional Network

## 1. Project Overview

This project implements an IoT network scheduling system using a Graph Convolutional Network (GCN).

The project is inspired by the scheduling problem addressed in the RobustGANTT framework. The current implementation focuses on a **GCN-based baseline** rather than the complete RobustGANTT Transformer-GNN architecture.

The main objective is to learn scheduling decisions for nodes in an IoT network.

Each node at each time slot is classified into one of three actions:

- **OFF** — the node is inactive.
- **TAG** — the node performs a tag-related operation.
- **CG** — the node acts as a carrier/gateway operation.

The implementation uses an **OR-Tools CP-SAT scheduler** to generate valid reference schedules. These schedules are then used as labels to train the GCN.

The trained GCN is finally tested on previously unseen IoT network topologies.

---

# 2. Overall Implementation

The complete implementation follows this pipeline:

```text
IoT Network Topology
        |
        v
Graph Construction
        |
        v
OR-Tools CP-SAT Scheduler
        |
        v
Reference Scheduling Decisions
        |
        v
Dataset Generation
        |
        v
Dataset Validation
        |
        v
GCN Training
        |
        v
Trained GCN Model
        |
        v
Held-Out Test Graphs
        |
        v
GNN Inference
        |
        v
OFF / TAG / CG Predictions
        |
        v
Actual vs Predicted Comparison
        |
        v
Visualization
```

---

# 3. Project Structure

```text
rgnn/
│
├── core/
│   └── generate_greedy/
│       │
│       ├── run_schedule.py
│       ├── schedule.py
│       ├── topology.py
│       ├── visualization.py
│       │
│       ├── generate_dataset.py
│       ├── dataset.pt
│       ├── dataset_summary.json
│       ├── validate_dataset.py
│       │
│       ├── train_gnn.py
│       ├── best_gcn_model.pt
│       ├── training_history.json
│       │
│       ├── generate_large_dataset.py
│       ├── large_dataset.pt
│       ├── large_dataset_summary.json
│       ├── validate_large_dataset.py
│       │
│       ├── train_large_gnn.py
│       ├── best_large_gcn_model.pt
│       ├── large_gnn_training_history.json
│       │
│       ├── visualize_gnn_prediction.py
│       ├── run_gnn.py
│       ├── current_gnn_prediction.png
│       ├── gnn_prediction_visualization.png
│       │
│       ├── ROBUSTGANTT_ARCHITECTURE.md
│       │
│       └── graphs/
│           ├── graph-1.json
│           ├── graph-2.json
│           ├── ...
│           └── graph-12.json
│
└── README.md
```

---

# 4. Technologies Used

## Programming Language

- Python 3

## Main Libraries

- PyTorch
- PyTorch Geometric
- NetworkX
- OR-Tools
- NumPy
- Scikit-learn
- Matplotlib

## Algorithms / Techniques

- Graph representation
- Graph Convolutional Network (GCN)
- OR-Tools CP-SAT scheduling
- Supervised learning
- Graph-level train/validation/test splitting
- Multi-class classification

---

# 5. Scheduling Problem

The IoT network is represented as a graph.

```text
G = (V, E)
```

where:

- `V` represents IoT nodes.
- `E` represents communication relationships between nodes.

Each node can have one or more tags that need to be scheduled.

For every node and time slot, the scheduler determines an action:

```text
OFF
TAG
CG
```

The objective of the scheduling stage is to generate a valid schedule while satisfying the required scheduling constraints.

---

# 6. OR-Tools Scheduler

The project uses the OR-Tools CP-SAT solver to generate reference schedules.

OR-Tools acts as the scheduling/optimization component.

It receives:

```text
IoT topology
+
node/tag information
+
scheduling constraints
```

and produces:

```text
Node × Time Slot Schedule
```

Example:

```text
          Slot 0   Slot 1   Slot 2

Node 0       CG       CG       CG
Node 1       CG       CG      OFF
Node 2       CG       CG      OFF
Node 3      TAG      TAG      TAG
Node 4      TAG      TAG       CG
...
```

These OR-Tools schedules are used as the reference labels for machine learning.

---

# 7. Dataset Generation

A large synthetic dataset was generated using different IoT network topologies.

The dataset contains:

- **500 graph samples**
- **4,109 active nodes**
- **3,427 tags**
- **1,238 schedule slots**
- **10,127 node-slot decisions**

The generated topology types include:

- Random / Erdos-Renyi
- Geometric
- Realistic Geometric
- Lattice / Grid

The scheduler successfully generated normal CP-SAT solutions for the 500 stored graphs.

Some generation attempts can fail because certain randomly generated topologies may not produce a valid scheduling instance. Such failed attempts are skipped rather than inserted into the final dataset.

---

# 8. Dataset Labels

Every node-slot decision has one of three labels.

| Label | Action | Meaning |
|---:|---|---|
| 0 | OFF | Node is inactive |
| 1 | TAG | Node performs a tag operation |
| 2 | CG | Node performs a carrier/gateway operation |

The dataset contains:

```text
OFF = 5,147
TAG = 3,427
CG  = 1,553
```

Total:

```text
5,147 + 3,427 + 1,553 = 10,127
```

The number of TAG operations corresponds to the number of hosted tags in the generated schedules.

---

# 9. Dataset Validation

Before training, the generated dataset is validated.

The validation checks include:

- Missing values
- NaN values
- Infinite values
- Invalid graph edges
- Invalid labels
- Schedule dimensions
- TAG operation consistency
- Number of hosted tags
- Dataset structure

Run:

```powershell
cd C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy

python validate_large_dataset.py
```

The dataset should pass all validation checks before training.

---

# 10. GCN Model

The machine learning component is a Graph Convolutional Network.

The GCN receives node-level features and graph connectivity.

The current input features are:

```text
1. Node degree
2. Number of hosted tags
3. X position
4. Y position
5. Slot fraction
```

The model learns relationships between neighboring nodes using graph convolution.

The final layer produces three class predictions:

```text
OFF
TAG
CG
```

Conceptually:

```text
Node Features
      |
      v
Graph Convolution
      |
      v
Hidden Representation
      |
      v
Graph Convolution
      |
      v
Classification Layer
      |
      v
OFF / TAG / CG
```

---

# 11. Training Procedure

The large dataset is divided at the graph level.

```text
Total graphs = 500

Training   = 350 graphs
Validation = 75 graphs
Testing    = 75 graphs
```

The split uses:

```text
Random Seed = 42
```

Graph-level splitting is important because nodes/slots from the same graph should not be distributed across training and testing.

The model is trained using:

- Adam optimizer
- Learning rate = `0.001`
- Maximum epochs = `100`
- Early stopping = `15` epochs
- Weighted cross-entropy loss

Class weights are used because the three classes are not equally distributed.

---

# 12. Train the GCN

To train the large GCN:

```powershell
cd C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy

python train_large_gnn.py
```

The trained model is saved as:

```text
best_large_gcn_model.pt
```

Training information is stored in:

```text
large_gnn_training_history.json
```

---

# 13. GCN Evaluation Results

The trained GCN was evaluated on 75 previously unseen graph topologies.

The final test results were:

| Metric | Result |
|---|---:|
| Test Accuracy | **57.85%** |
| Macro-F1 | **53.16%** |

Class-wise F1 scores:

| Class | F1 |
|---|---:|
| OFF | 0.6508 |
| TAG | 0.6114 |
| CG | 0.3327 |

The model successfully predicts all three classes on the held-out test set.

---

# 14. Confusion Matrix

The test confusion matrix was:

```text
                 Pred OFF   Pred TAG   Pred CG

Actual OFF          397        121        100

Actual TAG          122        306         77

Actual CG            83         69         82
```

This shows that the model can distinguish between the three scheduling actions, although CG remains the most difficult class.

---

# 15. Running the OR-Tools Scheduler

To run the original scheduling demonstration:

```powershell
cd C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy

python run_schedule.py graphs\graph-1.json
```

This runs the scheduler on:

```text
graphs\graph-1.json
```

The output displays the generated topology information and schedule.

---

# 16. Running the GNN Demonstration

The project includes:

```text
run_gnn.py
```

This script performs inference using the already trained model.

It does **not retrain the model**.

It loads:

```text
best_large_gcn_model.pt
large_dataset.pt
```

and selects a graph from the held-out test set.

The demonstration currently uses:

```text
Graph: synth_0099
Seed: 42
Nodes: 10
Tags: 13
Slots: 3
```

Run:

```powershell
cd C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy

python run_gnn.py
```

---

# 17. GNN Demonstration Output

The demonstration compares the OR-Tools schedule with the GNN prediction.

### Actual OR-Tools Schedule

```text
          Slot 0   Slot 1   Slot 2

Node 0       CG       CG       CG
Node 1       CG       CG      OFF
Node 2       CG       CG      OFF
Node 3      TAG      TAG      TAG
Node 4      TAG      TAG       CG
Node 5      TAG      TAG      TAG
Node 6      TAG      TAG      OFF
Node 7      TAG      OFF      OFF
Node 8      TAG      OFF      OFF
Node 9       CG       CG      TAG
```

### GNN Predicted Schedule

```text
          Slot 0   Slot 1   Slot 2

Node 0      TAG      OFF      OFF
Node 1      TAG       CG       CG
Node 2      TAG       CG      OFF
Node 3      TAG      TAG      TAG
Node 4      TAG      TAG      OFF
Node 5      TAG      TAG      TAG
Node 6      TAG       CG       CG
Node 7      TAG      TAG      OFF
Node 8      TAG       CG       CG
Node 9      TAG      TAG      TAG
```

For this demonstration:

```text
Total decisions = 30
Correct predictions = 16
Incorrect predictions = 14

Demo accuracy = 53.33%
```

This is the accuracy of the single demonstration graph and should not be confused with the overall test accuracy of 57.85%.

---

# 18. Visualization

The GNN demonstration generates a visualization comparing:

```text
Actual OR-Tools Schedule
          vs
GNN Predicted Schedule
```

The visualization includes:

- IoT graph topology
- Node IDs
- Graph edges
- Schedule slots
- Actual node actions
- Predicted node actions

Expected output files:

```text
current_gnn_prediction.png
```

and/or:

```text
gnn_prediction_visualization.png
```

They are generated in:

```text
core/generate_greedy/
```

To open the visualization in Windows:

```powershell
cd C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy

start current_gnn_prediction.png
```

If the filename is different, list the generated PNG files:

```powershell
dir *.png
```

---

# 19. Important Difference Between OR-Tools and GNN

The project contains two different components.

## OR-Tools

OR-Tools is used to generate the reference schedule.

```text
Topology
   ↓
Constraints
   ↓
OR-Tools CP-SAT
   ↓
Valid Reference Schedule
```

## GCN

The GCN learns from those schedules.

```text
Topology + Features
        ↓
       GCN
        ↓
OFF / TAG / CG
```

Therefore:

> OR-Tools generates the reference scheduling labels, while the GCN learns to approximate those scheduling decisions.

The GCN is **not currently used to replace the OR-Tools optimizer with a guaranteed constraint-valid schedule**.

---

# 20. Important Limitation

The current GCN makes node-slot predictions independently.

Therefore, its output can contain scheduling inconsistencies.

For example, in the demonstration:

```text
Actual TAG operations     = 13
Predicted TAG operations  = 18
Difference                = +5
```

This means the GCN prediction is not guaranteed to satisfy every global scheduling constraint.

This is an expected limitation of the current baseline implementation.

---

# 21. Relation to RobustGANTT

The project is inspired by the RobustGANTT scheduling framework.

The current implementation does **not** claim to implement the complete RobustGANTT Transformer-GNN architecture.

The current implementation uses:

```text
OR-Tools
   +
GCN
```

as a baseline scheduling-learning pipeline.

The full RobustGANTT architecture uses a more advanced Transformer-GNN approach with dynamic/autoregressive scheduling.

---

# 22. Future Work

Possible future improvements include:

1. Implementing the complete RobustGANTT Transformer-GNN architecture.

2. Using dynamic scheduling features.

3. Adding autoregressive prediction across time slots.

4. Recomputing node features after tags are served.

5. Adding constraint-aware inference.

6. Improving CG classification.

7. Comparing GCN performance with Transformer-GNN performance.

8. Evaluating scheduling efficiency against the OR-Tools reference schedules.

---

# 23. Quick Start

If the dataset and trained model already exist, the simplest demonstration is:

```powershell
cd C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy

python run_gnn.py
```

For the OR-Tools scheduler:

```powershell
python run_schedule.py graphs\graph-1.json
```

---

# 24. Complete Execution Order

For reproducing the complete implementation from the beginning:

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

If PyTorch/PyTorch Geometric are not installed, install the compatible versions required by the environment.

---

### Step 2 — Generate the dataset

```powershell
python generate_large_dataset.py
```

This creates:

```text
large_dataset.pt
large_dataset_summary.json
```

---

### Step 3 — Validate the dataset

```powershell
python validate_large_dataset.py
```

---

### Step 4 — Train the GCN

```powershell
python train_large_gnn.py
```

This creates:

```text
best_large_gcn_model.pt
large_gnn_training_history.json
```

---

### Step 5 — Run GNN inference

```powershell
python run_gnn.py
```

---

### Step 6 — View the visualization

```powershell
start current_gnn_prediction.png
```

---

# 25. Final Project Summary

The implemented system demonstrates a complete machine-learning-based IoT scheduling pipeline.

The system first creates different IoT network topologies and represents them as graphs. OR-Tools CP-SAT is then used to generate reference schedules satisfying the scheduling formulation. These schedules form the supervised learning dataset.

A Graph Convolutional Network is trained using node, topology, and time-slot features. The model learns to classify scheduling actions into OFF, TAG, and CG.

The trained model is evaluated on previously unseen network topologies.

The current implementation achieved:

```text
Test Accuracy : 57.85%
Macro-F1       : 53.16%
```

The system also provides a demonstration script that compares the reference OR-Tools schedule with the GNN prediction and generates a visualization.

The implementation therefore demonstrates:

```text
IoT Graph Generation
        ↓
Graph Representation
        ↓
Constraint-Based Scheduling
        ↓
Dataset Creation
        ↓
GCN Training
        ↓
Model Evaluation
        ↓
GNN Inference
        ↓
Schedule Comparison
        ↓
Visualization
```

This forms the current **GCN baseline implementation** for the IoT scheduling project.
>>>>>>> 321446bbac8a9085c5211c7f7d18ccf7ff8cca1b
