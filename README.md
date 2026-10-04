
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
