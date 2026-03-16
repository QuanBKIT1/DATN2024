# CLAUDE.md — AI Assistant Guide for DATN2024

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository.

---

## Project Overview

This is a research project for **skeleton-based Word-Level Sign Language Recognition (SLR)** on the WLASL2000 dataset (2000 American Sign Language classes, ~21K videos).

The pipeline:
1. Extract whole-body skeleton keypoints from raw videos using RTMPose ONNX models (`genpose.py`)
2. Train/evaluate graph convolutional networks on skeleton sequences (`training.py`)

Best published result: **56.46% Top-1 / 87.94% Top-5** accuracy using RTMW-l+ + ST-GCN++ + 31 keypoints + NLA (ε=0.3).

---

## Repository Structure

```
DATN2024/
├── configs/
│   ├── ctr-gcn/config.yaml       # CTR-GCN training configuration
│   └── stgcn-pp/config.yaml      # ST-GCN++ training configuration
├── checkpoints/                  # Pre-trained model checkpoint YAMLs
├── data/
│   ├── wlasl_2000.json           # Dataset metadata
│   ├── wlasl_class_list.txt      # 2000 sign class names (index → gloss)
│   └── wlasl_word_embeddings.pkl # fastText 300-dim embeddings per class
├── feeder/
│   ├── feeder.py                 # PyTorch Dataset — loads skeleton .npy files
│   └── tools.py                  # Augmentation helpers (choose, shift, move, mirror)
├── graph/
│   └── graph.py                  # Skeleton graph topology (27/31/49 keypoint layouts)
├── model/
│   ├── ctrgcn/ctrgcn.py          # Channel-wise Temporal-Relational GCN
│   └── stgcn_pp/
│       ├── stgcn_pp.py           # ST-GCN++ model
│       └── utils/
│           ├── gcn.py            # Unit GCN layer
│           └── tcn.py            # Temporal convolution modules (MSTCN)
├── utils/
│   ├── loss.py                   # LabelSmoothCE with optional NLA (word embedding similarity)
│   ├── tools.py                  # Data processing (downsample, pad, slice, augment)
│   ├── pre_processing.py         # RTMPose input preprocessing
│   ├── post_processing.py        # RTMPose output postprocessing
│   ├── inference.py              # Batch inference utilities
│   ├── configs.py                # YAML config helpers
│   ├── webcam_demo.ipynb         # Live webcam demo
│   ├── gen_word_embedding_wlasl.ipynb  # Word embedding generation
│   └── count_params.ipynb        # Model parameter analysis
├── visualize/                    # Output images/GIFs (not tracked heavily)
├── genpose.py                    # Pose extraction pipeline (ONNX → .npy keypoints)
├── training.py                   # Main entry point: loads config → runs Processor
├── processor.py                  # Training/testing engine (396 lines)
├── requirements.txt              # Python pip dependencies
└── README.md                     # Project documentation with results tables
```

---

## Technology Stack

| Component | Library / Version |
|-----------|-------------------|
| Deep learning | PyTorch 2.3.1 |
| Pose inference | ONNX Runtime 1.18.1 |
| Computer vision | OpenCV 4.10.0.84 |
| Numerics | NumPy 1.26.4 |
| Configuration | PyYAML 6.0.1 |
| Data | Pandas 2.2.2 |
| Graph math | NetworkX 3.3 |
| Visualization | Matplotlib 3.9.0 |
| Progress bars | tqdm 4.66.4 |

Python 3.x required. Install everything with:
```bash
pip install -r requirements.txt
```

---

## Key Concepts & Data Format

### Skeleton Tensor Format: `(N, C, T, V, M)`
- **N** — batch size
- **C** — channels: `[x, y, confidence]` (3 values per keypoint)
- **T** — temporal frames (variable length, padded/sampled to fixed window, typically ≤300)
- **V** — vertices / keypoints (`27`, `31`, or `49` depending on layout)
- **M** — persons per frame (usually `1`)

### Keypoint Layouts
Defined in `graph/graph.py`. Three options:
- `"keypoint-27"` — 27 upper-body + hand joints
- `"keypoint-31"` — 31 joints (extends 27 with extra finger joints)
- `"keypoint-49"` — 49 whole-body joints

Layout is set in the config YAML under `graph_layout`.

### Pose Estimators Supported
| Model | Input Size | AP | Notes |
|-------|-----------|-----|-------|
| RTMPose-l | 256×192 | 61.1 | Fastest |
| HRNet-w48-Dark | 384×288 | 66.1 | Good accuracy |
| RTMW-l+ | 384×288 | 70.1 | Best accuracy (recommended) |

ONNX weights are **not** included in the repository — download separately.

---

## Model Architectures

### ST-GCN++ (`model/stgcn_pp/stgcn_pp.py`)
- Stacked `STGCNBlock` modules combining spatial GCN + temporal TCN (MSTCN)
- Configurable number of stages and channels
- Entry class: `Model`

### CTR-GCN (`model/ctrgcn/ctrgcn.py`)
- Channel-wise Topology Refinement GCN
- 10 residual GCN layers (l1–l10) with adaptive adjacency matrix learning
- Entry class: `Model`

Both models share the same interface: they accept `(N, C, T, V, M)` tensors and output class logits of shape `(N, num_class)`.

---

## Natural Language Awareness (NLA)

Controlled by `use_nla: True/False` and `nla_epsilon: 0.0–1.0` in config.

When enabled, `utils/loss.py` (`LabelSmoothCE`) replaces one-hot hard labels with soft labels derived from **fastText word embedding cosine similarity** between class glosses. This encodes semantic relatedness between signs into the loss.

Word embeddings are pre-computed and stored in `data/wlasl_word_embeddings.pkl`.

---

## Configuration System

All training is controlled through YAML config files. Key fields:

```yaml
# configs/stgcn-pp/config.yaml (example excerpt)
Experiment_name: stgcn-pp
phase: train               # "train" or "test"

train_data_path: path/to/train.npy
train_label_path: path/to/train_labels.pkl
val_data_path: path/to/val.npy
val_label_path: path/to/val_labels.pkl
test_data_path: path/to/test.npy
test_label_path: path/to/test_labels.pkl

work_dir: ./work_dir/stgcn-pp/   # outputs: logs, checkpoints, CSV
save_epoch: 5

model: model.stgcn_pp.stgcn_pp.Model
model_args:
  num_class: 2000
  graph_args:
    layout: keypoint-31

graph_layout: keypoint-31

batch_size: 64
num_epoch: 100
optimizer: SGD
base_lr: 0.1
momentum: 0.9
weight_decay: 0.0004

label_smoothing: 0.1
use_nla: True
nla_epsilon: 0.3
```

To switch to test mode: change `phase: train` → `phase: test` and provide `--weight`.

---

## Common Commands

### Training
```bash
# ST-GCN++
python training.py --config configs/stgcn-pp/config.yaml

# CTR-GCN
python training.py --config configs/ctr-gcn/config.yaml

# Resume from checkpoint
python training.py --config configs/stgcn-pp/config.yaml --weight path/to/checkpoint.pt
```

### Testing (set phase: test in config first)
```bash
python training.py --config configs/stgcn-pp/config.yaml --weight path/to/best_model.pt
```

### Pose Extraction
```bash
python genpose.py \
    --onnx-path ./onnx_model/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx \
    --video-data-path ./WLASL2000 \
    --device cpu \
    --save-path ./keypoints_rtmpose_wholebody/
```

Use `--device cuda` for GPU acceleration during pose extraction.

### Webcam Live Demo
```bash
jupyter notebook utils/webcam_demo.ipynb
```

---

## Output Files

Training writes to `work_dir/` (configured in YAML):
```
work_dir/<experiment_name>/
├── epoch_info_training.csv     # Per-epoch Top-1, Top-5, loss
├── log.txt                     # Full training log
└── epoch_<N>_model.pt          # Checkpoints (saved every save_epoch epochs)
```

---

## Code Conventions

- **Model entry class** must be named `Model` (imported dynamically via `model_args` in config).
- **Config loading**: `utils/configs.py` handles YAML parsing and CLI override merging.
- **Tensor layout**: Always `(N, C, T, V, M)` — not `(N, T, V, C)`. Keep this consistent when adding new models.
- **Adjacency matrix**: Passed as `self.A` in graph module; models accept it in `__init__` via `graph_args`.
- **Loss function**: Always `LabelSmoothCE` from `utils/loss.py`. Do not replace with `nn.CrossEntropyLoss` without considering NLA.
- **Checkpoints**: Saved as `torch.save(model.state_dict(), path)`. Load with `model.load_state_dict(torch.load(path))`.
- **No test suite**: Testing is config-driven (set `phase: test`). There are no pytest/unittest files.
- **Notebooks**: Kept in `utils/` for interactive exploration. Not part of the training pipeline.

---

## Data Files Not Included in Repo

The following large files must be obtained separately:
- **WLASL2000 video dataset** — Download from the official WLASL repository
- **Skeleton `.npy` files** — Generated by running `genpose.py` on the video dataset
- **Label `.pkl` files** — Generated alongside the `.npy` files
- **ONNX model weights** — RTMPose/HRNet/RTMW ONNX files for pose extraction

The `data/` directory contains only metadata and pre-computed word embeddings (~small files).

---

## Extending the Project

### Adding a New Model
1. Create `model/<your_model>/<your_model>.py` with a class named `Model`
2. Ensure `Model.__init__` accepts `num_class`, `graph_args` (at minimum)
3. Ensure `Model.forward(x)` accepts `(N, C, T, V, M)` and returns `(N, num_class)`
4. Add a new config YAML pointing to your model: `model: model.<your_model>.<your_model>.Model`

### Adding a New Keypoint Layout
1. Edit `graph/graph.py` — add a new layout name and its edge definitions
2. Add the layout to `feeder/feeder.py` if the feeder needs layout-specific handling
3. Update the `graph_layout` field in your config YAML

### Modifying NLA
The semantic similarity soft-label logic lives entirely in `utils/loss.py`. Word embeddings are loaded once in `processor.py` and passed to the loss function.
