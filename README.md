# MultiLabel-PPI

Official implementation of **"MultiLabel-PPI: Riemannian Pair Modeling in ESM2-Enhanced Hyperbolic
Graph Networks for Multi-Label Protein–Protein Interaction Prediction."**

Multi-label PPI prediction over 7 STRING interaction types on SHS27K / SHS148K.
See the paper for method, datasets, and results; this README covers setup and how to run.

## Environment

Python 3.12.13, PyTorch 2.4.0 (CUDA 12.4), single NVIDIA L4 GPU.

```bash
pip install --upgrade --force-reinstall dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install torch-geometric torchdata==0.9.0 gensim fair-esm
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

## Data

**Structure features** — download and unzip into `features/`:
- SHS27K: https://drive.google.com/file/d/1SEplMBH36521XsG0yIDLY7X5xRaN7Ekb/view?usp=sharing
- SHS148K: https://drive.google.com/file/d/1Lqyg05aTbXYTb-uXpl3F36TfY7VTmk-B/view?usp=sharing

**ESM2 embeddings** — precompute once per dataset:

```bash
python3 scripts/precompute_esm2.py \
  -i data/protein.SHS148k.sequences.dictionary.tsv \
  -o features/148K_esm2_t12_35M_mean.pt \
  --pooling mean \
  --model facebook/esm2_t12_35M_UR50D
# SHS27K: -i data/protein.SHS27k.sequences.dictionary.tsv -o features/27K_esm2_t12_35M_mean.pt
```

## Run

The shell drivers take positional args `<DATASET> <SPLIT> <EPOCHS>`, loop over the seeds, run in
**read mode** on the fixed split `data/<DATASET>_<SPLIT>.json`, and aggregate mean ± std.

Main comparison (Ours vs. HI-PPI baseline) on SHS27K / BFS:

```bash
SEEDS="1 5 10 42 60" ESM_MODEL=esm2_t12_35M ESM_POOLING=mean ESM_ZSCORE=true \
  bash scripts/run_paper_compare.sh 27K bfs 100
```

Ablation (leave-one-out over components; same interface):

```bash
SEEDS="1 5 10 42 60" ESM_MODEL=esm2_t12_35M ESM_POOLING=mean ESM_ZSCORE=true \
  bash scripts/run_component_ablation.sh 27K bfs 100
```

Swap `27K`→`148K` and `bfs`→`dfs` for the other settings.
