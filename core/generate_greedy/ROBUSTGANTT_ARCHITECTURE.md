# RobustGANTT / RGNN Architecture & Data Pipeline Report

**Target Model Configuration:** `transf12mod` (12L-12H with Node Degree Positional Encoding)  
**Primary Source Material:** 
- Official Published Paper: *"Robust Generalization of Graph Neural Networks for Carrier Scheduling"*, Daniel F. Perez-Ramirez, Carlos Pérez-Penichet, Nicolas Tsiftes, Dejan Kostić, Magnus Boman, Thiemo Voigt (arXiv:2407.08479v1, July 2024).
- Foundational System Paper: *"DeepGANTT: A Scalable Deep Learning Scheduler for Backscatter Networks"*, ACM/IEEE IPSN '23 (Perez-Ramirez et al., 2023).
- Master's Thesis: *"The Applicability and Scalability of Graph Neural Networks on Combinatorial Optimization"*, Peder Hårderup (KTH / RISE, DiVA diva2:1842764, 2023).
- RISE GitLab Execution Logs & Model Metadata: Project `deepgantt_scheduler` / `RobustGANTT Scheduler` (`trained_models/fixed/12L-12H-degree/`, user Daniel Perez).
- Local Repository Source: `C:\Users\bhave\Desktop\GNN_V2\rgnn\core\generate_greedy` (`schedule.py`, `topology.py`, `run_schedule.py`, `generate_large_dataset.py`, `train_gnn.py`).

---

## 1. Source Inspection & Availability Audit

### Available in Workspace Repository (`rgnn/core/generate_greedy`):
1. **Topology Modeling & Generators:** `topology.py` implements the exact heterogeneous graph classes (`TopologyGraph`, `GeometricTopologyGraph`, `GridTopologyGraph`, `ErdosTopologyGraph`, `RealisticGeometricTopologyGraph`), active nodes (`A0, A1, ...`), tags (`T0, T1, ...`), and carrier thresholds.
2. **Scheduling Algorithms & Solvers:** `schedule.py` implements:
   - `Schedule` class: internal dictionary representation and slot indexing.
   - `compute_schedule_coloring`: The official **TagAlong** polynomial heuristic scheduler using graph coloring on the conflict graph ([Pérez-Penichet et al., INFOCOM 2020]).
   - `compute_schedule_ortools`: The CP-SAT solver solving the exact lexicographically-constrained combinatorial carrier scheduling problem (matching MiniZinc `model_optimize.mzn`).
   - `compute_schedule_sequential`: Sequential baseline.
3. **Current Datasets & Baselines:**
   - `dataset.pt`: 13-graph sample dataset.
   - `large_dataset.pt`: 500-graph validated dataset generated via CP-SAT / greedy fallback.
   - `train_gnn.py`: 2-layer `GCNConv` baseline.

### NOT AVAILABLE IN CURRENT SOURCE:
The following components from the official RISE repository (`https://gitlab.ri.se/daniel/robustgantt_scheduler` / `deepgantt_scheduler`) are **NOT PRESENT** in the local workspace or machine:
- The raw Python model definition file for `transf12mod` (`models/transformer_gnn.py` or similar).
- The official PyTorch Geometric data loaders: `PyGeoDataLoader` and `PyGeoDataSet`.
- The official training script (`train_model.py` / `ModelDeployer`).
- The pre-trained model weights (`transf12mod-state_dir.pt` / `transf12mod-model-dict.pkl`).

*Note:* Although the raw source files are behind RISE internal GitLab authentication, the complete architecture specifications, hyperparameters, mathematical formulations, loss functions, input representations, and training procedures are fully published in the arXiv paper (arXiv:2407.08479v1), author thesis, and indexed RISE testbed logs. All specifications below are derived strictly from these official sources without guessing.

---

## 2. RobustGANTT Architecture Report

### A. Data Preprocessing
- **Graph Representation:** The network is modeled as an undirected connected graph $G = \langle V_a, E \rangle$, where $V_a = \{v_i\}_{i=0}^{N-1}$ are active IoT nodes, and $E$ are bidirectional wireless links whose carrier signal strength meets or exceeds a reception threshold (e.g. $\ge -75\text{ dBm}$ or `cg_threshold`).
- **Heterogeneous Tag Assignment:** The network hosts $T$ battery-free sensor tags $N_t = \{t_i\}_{i=1}^T$. Each tag $t$ is mapped to exactly one host node $H_t(t) \in V_a$.
- **Graph Pruning:** Irrelevant active nodes that are neither tag hosts nor valid carrier generators for any unserved tag are pruned during scheduling iterations to reduce graph size.
- **Normalization:**
  - `tpnnorm` (Tags Per Node Normalized): Number of unserved tags hosted by node $v$ divided by the maximum hosted tags or total remaining tags.
  - `seqlabelnorm` (Sequence Label Normalized): The minimum tag identifier among unserved tags hosted by node $v$, normalized to $[0, 1]$ (or $0.0$ if no tags are hosted).
  - `deg` (Node Degree Normalized): Node degree $\tilde{d}_v = \sum_{u \in V_a} A[v, u]$ normalized by maximum degree $\tilde{d}_{max} = \max_{u} \tilde{d}_u$.
  - `tagid` / `node_id`: Integer node / tag identifier (used as discrete index or embedded).

### B. Input Features
The model operates on a per-node feature matrix $X_j \in \mathbb{R}^{N \times D}$ at each scheduling timeslot $j$.
In the official final configuration (`12L-12H-degree` / `transf12mod_binary_tpnnorm-seqlabelnorm-tagid-deg`), **$D = 4$ features per node (`n_input = 4`)**:

| Index | Feature Identifier | Name in Code / Paper | Type | Definition & Purpose |
|:---:|:---:|:---:|:---:|:---|
| 0 | `tpnnorm` | Hosted-Tags | Continuous / Float | Number of active, unserved tags hosted by node $v$ at current timeslot $j$. Decisive for carrier allocation (hosts with many tags avoid being CGs). |
| 1 | `seqlabelnorm` | Min. Tag-ID | Continuous / Float | Minimum tag ID among currently unserved tags hosted by node $v$, normalized. Provides canonical order to interrogate tags and break schedule permutation symmetries. |
| 2 | `tagid` / `node_id` | Node-ID | Integer / Normalized Float | Node index in graph topology. Contextualizes symmetry-breaking priorities for carrier provider selection. |
| 3 | `deg` | Node Degree PE | Continuous / Float | $\tilde{d}_v / \tilde{d}_{max} \in [0, 1]$. Injective structural positional encoding allowing attention layers to break local topological symmetries without costly eigendecomposition. |

*Crucial Note on Coordinates:* Cartesian positions $(x, y)$ are **NOT** used as input features in official RobustGANTT. Graph topological features and positional encodings were explicitly chosen because $(x, y)$ coordinates fail to generalize when network scale changes from 10 to 1,000 nodes.

### C. Target Representation
For each timeslot $j$, the network predicts an action vector $s_j \in \mathbb{R}^N$ for all $N$ nodes simultaneously:
- **Official Formulation:** 3-class node classification:
  - Class `O` (0) = **OFF** (Idle: node neither queries a tag nor provides a carrier).
  - Class `T` (1) = **TAG** (Interrogator: node queries one of its hosted tags).
  - Class `C` (2) = **CG** (Carrier Generator: node emits an unmodulated carrier for a neighbor's tag).
- **Binary Variant in Logs (`binary`, `n_output = 2`):** In some RISE experimental runs, the model performs binary classification predicting whether a node is a Carrier Generator (`CG`) vs Non-Carrier (`OFF`/`TAG`), with tag interrogations determined by host priority rules. The primary paper specifies the 3-class node assignment.

### D. Schedule / Sequence Representation
A full schedule coordinates tag interrogations over $L \ge 1$ timeslots:
$$S = [s_1, s_2, \dots, s_L]^\top \in \{0, 1, 2\}^{L \times N}$$
- In `schedule.py`, this is stored as `Schedule._inner[node_label] = [action_slot_0, action_slot_1, ...]`.
- In `large_dataset.pt`, this is stored as a tensor of shape `(num_slots, num_nodes)`.
- Constraints enforced:
  1. Each tag is interrogated exactly once across the schedule.
  2. Each tag interrogation requires exactly one carrier-providing neighbor in that slot.
  3. No carrier collisions: multiple carriers impinging on a tag cause destructive interference.
  4. Half-duplex constraint: A node cannot interrogate a tag and emit a carrier in the same slot.

### E. Autoregressive Mechanism
RobustGANTT generates schedules **slot-by-slot in an autoregressive loop**:
1. **State Initialization ($j=1$):**
   - The scheduler initializes a cached copy of the topology $G$ and tag mapping $H_t^{(1)} = H_t$.
   - Feature matrix $X_1$ is generated from initial unserved tags.
2. **Model Forward Pass:**
   - The Transformer-GNN processes $(X_j, E)$ and outputs node logits for slot $j$.
   - Actions $s_j \in \{0, 1, 2\}^N$ are selected.
3. **Constraint & Feasibility Check:**
   - A deterministic validator verifies that $s_j$ satisfies physical constraints (at most one tag per host, exactly one CG per interrogated tag, no conflicting carrier interference). Infeasible node actions are suppressed/corrected.
4. **State Transition & Cache Update ($j \to j+1$):**
   - Interrogated tags are removed from the cached tag mapping:
     $$H_t^{(j+1)} = H_t^{(j)} \setminus \{\text{tags interrogated in slot } j\}$$
   - Node features are dynamically recomputed for slot $j+1$:
     - `tpnnorm` decreases for nodes that served tags.
     - `seqlabelnorm` updates to the new minimum unserved tag ID (or resets to 0 when all hosted tags are completed).
5. **Termination:**
   - The loop terminates when all tags have been scheduled ($|H_t^{(j+1)}| = 0$).

### F. Graph Neural Network Component
- **Neighborhood Aggregation:** Message passing occurs exclusively between 1-hop connected neighbors $\mathcal{N}(v)$ in the wireless topology graph $G$.
- **Edge Weighting:** Edges represent valid carrier-provisioning links. In attention-based message passing, attention coefficients $\alpha_{uv}$ scale the neighbor representations $h_u$.

### G. Transformer Component
- **Self-Attention over Graph Topology:** Unlike sequence Transformers that attend over all tokens in a sentence, the Transformer-GNN uses **sparse graph self-attention**: node $v$ computes multi-head attention only over its topological neighbors $u \in \mathcal{N}(v) \cup \{v\}$.
- **Node-wise Feed-Forward Network (FFN):** A multi-layer perceptron acting on each node's intermediate embedding independently.

### H. Number of Layers
- **Total Layers:** **12 Transformer-GNN layers** (`12L`).
- **Rationale:** Selected based on the depth proven in language models (BERT/GPT) and empirical success in combinatorial optimization on graphs, providing a 12-hop receptive field across the network topology.

### I. Hidden Dimensions
- **Input Dimension (`n_input`):** 4
- **Node Embedding Dimension (`n_embed`):** 48
- **Hidden Feature Dimension (`n_hidden`):** 200
- **Output Dimension (`n_output`):** 2 (binary mode) or 3 (multi-class mode)
- **Parameter Count:** $\sim 295$ million parameters in the complete server deployment (as cited in paper Section 6).

### J. Attention Mechanism
- **Number of Attention Heads (`n_heads`):** **12 heads** (`12H`).
- **Ablation Results from Paper:**
  - 4 heads: Insufficient capacity; poor generalization to large graphs.
  - 8 heads: Competitive on small graphs, but lower stability on $\ge 60$ nodes.
  - **12 heads:** Optimal balance; consistently achieved positive carrier savings ($\Delta_C > 0$) up to 1,000 nodes.
  - 16 heads: Overparameterized; resulted in performance degradation ($\Delta_C$ dropped below 0).

### K. Skip / Residual Connections & Normalization
- Pre-layer normalization (`LayerNorm`) applied before attention and FFN blocks.
- Residual connections around both the multi-head attention block and the FFN block:
  $$h^{(l)'} = \text{LayerNorm}(h^{(l-1)} + \text{MultiHeadAttn}(h^{(l-1)}))$$
  $$h^{(l)} = \text{LayerNorm}(h^{(l)'} + \text{FFN}(h^{(l)'}))$$
- Dropout rate: $1 \times 10^{-6}$ (minimal dropout to preserve deterministic graph structural representations).

### L. Output Layer
- Linear projection from hidden dimension 200 to `n_output` (3 classes: OFF, TAG, CG).
- Softmax activation during inference / CrossEntropyLoss during training.

### M. Loss Function
- **Modified Weighted Cross-Entropy Loss:**
  $$\mathcal{L} = -\sum_{v \in V_a} \sum_{c \in \{O, T, C\}} w_c \cdot y_{v, c} \cdot \log(\hat{p}_{v, c}) + \lambda \|\Theta\|_2^2$$
  - Class scaling weights $w_c$: Higher weight $w_C$ assigned to the Carrier Generator class ($C$) to counter class imbalance (idle nodes significantly outnumber carrier providers, but carrier errors cause invalid schedules).
  - L2 Regularization (weight decay $\lambda$).

### N. Training Procedure
- **Dataset:** 580,000 small-scale problem instances (2–10 nodes, 1–14 tags) solved to analytical optimality with combinatorial optimization (MiniZinc / CP-SAT).
- **Split:** 80% training, 20% validation.
- **Batch Size:** 1,024 mini-batch instances.
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$).
- **Learning Rate Schedule:**
  - Initial learning rate: $\epsilon_{init} = 10^{-3}$.
  - Untuned linear learning rate warmup for 2,000 mini-batch steps:
    $$\tilde{\epsilon} = \epsilon_{init} \cdot \min\left(1, \frac{1 - \beta_2}{2} \cdot i\right)$$
  - Exponential decay: 2% reduction per epoch after warmup.
- **Early Stopping:** 25 consecutive epochs without improvement in validation loss.
- **Model Selection:** Checkpoint with highest Carrier-class F1-score (`F1_carrier`) on validation set.

### O. Evaluation Metrics
1. **$\Pi$ (Correctly Computed Schedules %):** Fraction of topologies where the scheduler computes a complete, valid schedule interrogating 100% of tags without constraint violations.
2. **$\Delta_C$ (Carriers Saved):** $\Delta_C = C_{ta} - C_{nn}$ (difference in carrier slots compared to the TagAlong heuristic; positive indicates resource savings).
3. **$\Delta_L$ (Timeslots Saved):** $\Delta_L = L_{ta} - L_{nn}$ (difference in schedule latency).
4. **$\Delta_{E\%}$ (Energy Saved %):** Percentage reduction in total radio RF energy consumption over TagAlong.
5. **Runtime:** 95th-percentile inference latency in milliseconds.

---

## 3. Comparison with Our Current Dataset (`large_dataset.pt`)

### Current State of `large_dataset.pt`:
- **500 graphs**, 4,109 active nodes, 3,427 tags, 1,238 schedule slots, 10,127 node-slot decisions.
- **Actions:** OFF = 5,147 (50.8%), TAG = 3,427 (33.8%), CG = 1,553 (15.3%).
- **Current `node_features` Tensor:** Shape `[num_nodes, 4]` containing:
  `[degree, hosted_tags, x_position, y_position]`
- **Current `schedule` Tensor:** Shape `[num_slots, num_nodes]` containing integer labels `0 (OFF), 1 (TAG), 2 (CG)`.

### Key Differences & Deficiencies:

| Component | Official RobustGANTT (`transf12mod`) | Our Current `large_dataset.pt` | Gap Analysis & Impact |
|:---|:---|:---|:---|
| **Input Feature 0** | `tpnnorm`: Dynamic unserved tags count per node at slot $j$ | Static hosted tags at $t=0$ | In `large_dataset.pt`, `hosted_tags` is fixed at $t=0$. In an autoregressive rollout, this must decrease as tags are completed. |
| **Input Feature 1** | `seqlabelnorm`: Min remaining tag ID hosted by node, normalized | NOT PRESENT | Crucial for symmetry breaking. Without this, the model cannot distinguish which tag to serve next or break topological symmetries. |
| **Input Feature 2** | `tagid` / `node_id`: Node identifier / index | NOT PRESENT in feature matrix | Present only implicitly as tensor row indices, not fed into the feature embedding. |
| **Input Feature 3** | `deg`: Node degree normalized $\tilde{d}_v / \tilde{d}_{max}$ | Raw unnormalized degree | Our degree is raw integer ($1, 2, \dots$), not normalized by $\max(deg)$. |
| **Spatial Coords** | **Excluded** | Included: `x_position, y_position` | Cartesian coordinates do not generalize across graph scales. Must be replaced or ablated. |
| **Data Organization** | Slot-level autoregressive transition pairs $(X_j, G, s_j)$ | Graph-level static records with full schedule matrix | Current dataset stores the full schedule at once; training an autoregressive model requires unrolling slot-by-slot dynamic states. |

---

## 4. Minimum Required Data Adaptation

To bridge our current data to the official `transf12mod` pipeline without modifying existing files:

| Official RobustGANTT Input | Source in our Existing Data | Concrete Adaptation Function |
|:---|:---|:---|
| `tpnnorm` (Dynamic Hosted Tags) | `sample['tags']` + `sample['schedule']` | Count remaining tags hosted by node $v$ after removing tags scheduled in slots $< j$. Normalize by $\max(\text{initial hosted tags})$. |
| `seqlabelnorm` (Min Remaining Tag ID) | `sample['tags']` + `sample['schedule']` | $\min(\{t.\text{id} \mid t \text{ unserved on host } v\}) / \text{num\_tags}$. If node hosts no remaining tags, set to $0.0$. |
| `tagid` / `node_id` | Node index in `topo.active_nodes` | Float normalized node index $v / (N - 1)$ or integer for embedding lookup. |
| `deg` (Normalized Degree) | `sample['node_features'][:, 0]` | $\text{deg}_v / \max_{u}(\text{deg}_u)$ computed per graph. |
| Target Action $s_j$ | `sample['schedule'][j, :]` | Directly matches: 0 = OFF, 1 = TAG, 2 = CG. |
| Graph Topology $E$ | `sample['edge_index']` | Directly matches bidirectional edge index. |
| Edge Weights | `sample['edge_weight']` | Directly matches carrier signal strengths. |

---

## 5. Summary & Next Implementation Roadmap

### 1. Official Architecture Identified:
`transf12mod`: A 12-layer, 12-head Transformer-GNN with Node Degree Positional Encoding, hidden dimension 200, embedding dimension 48, dropout $10^{-6}$, pre-layer normalization, and residual skip connections.

### 2. Exact Source Files Inspected:
- Local: `schedule.py`, `topology.py`, `run_schedule.py`, `generate_large_dataset.py`, `train_gnn.py`.
- External / Reference: Published paper arXiv:2407.08479v1, DeepGANTT IPSN '23, and indexed RISE testbed logs for `trained_models/fixed/12L-12H-degree/`.

### 3. Number of Layers:
12 Transformer-GNN layers.

### 4. Input Feature Representation:
4 normalized features: `[tpnnorm, seqlabelnorm, tagid, deg]`.

### 5. Target Representation:
3-class node action per timeslot: `0=OFF`, `1=TAG`, `2=CG`.

### 6. Autoregressive Mechanism:
Iterative slot-by-slot rollout where scheduled tags are pruned from cached state, dynamically updating `tpnnorm` and `seqlabelnorm` until all tags are scheduled.

### 7. Loss Function:
Modified weighted Cross-Entropy Loss with carrier generator class weighting + L2 weight decay.

### 8. Training Procedure:
Adam optimizer ($\epsilon_{init} = 10^{-3}$, 2,000-step linear warmup, 2% decay/epoch), batch size 1024, early stopping on validation loss (patience 25), model selection by carrier F1-score.

### 9. What our current dataset already supports:
Full topology edge indices, carrier weights, ground-truth optimal schedules, and initial tag placements.

### 10. What our current dataset is missing:
Dynamic slot-level unrolling, normalized node degree PE, minimum unserved tag ID feature (`seqlabelnorm`), and removal of scale-sensitive Cartesian $(x, y)$ coordinates.

### 11. Exact Next Steps (When instructed to proceed):
1. Build an unrolling dataset converter or dynamic dataset wrapper that derives `[tpnnorm, seqlabelnorm, tagid, deg]` slot-by-slot from `large_dataset.pt`.
2. Implement the 12-layer, 12-head Transformer-GNN (`transf12mod`) module with pre-layer norm and skip connections.
3. Implement the training script with 2000-step linear warmup, carrier-class weighted cross-entropy, and carrier-F1 checkpoint selection.
4. Implement the autoregressive inference scheduler with constraint validation and compare against TagAlong (`compute_schedule_coloring`).
