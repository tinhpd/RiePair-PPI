#!/usr/bin/env bash
# Component ablation for hippi_riem_branch_gate.
# Removes one component at a time to measure each contribution.
#
# Ablation matrix:
#   hippi_riem_branch_gate  Full model         (reference)
#   hippi_riem_noesm        Disable ESM2       → seq_encoder=handcrafted
#   hippi_riem_nosym        Disable Sym Adj    → typed_edge_mode=original
#   hippi_riem_noskip       Disable Residual   → pair_head=riemann, riemann_residual=false
#   hippi_pair_gated        Disable Riem Head  → pair_head=gated
#
# Usage:
#   bash scripts/run_component_ablation.sh [DATASET] [SPLIT] [EPOCHS]
#
# Examples:
#   SEEDS=1,5,10,42,60 bash scripts/run_component_ablation.sh 27K bfs 100
#   ESM_ZSCORE=true ESM_POOLING=mean ESM_MODEL=facebook/esm2_t12_35M \
#     SEEDS=1,5,10,42,60 bash scripts/run_component_ablation.sh 148K bfs 100

set -euo pipefail

DATASET="${1:-27K}"
SPLIT="${2:-bfs}"
EPOCHS="${3:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEEDS="${SEEDS:-${SEED:-7}}"
PYTHON="${PYTHON:-python3}"
ESM_MODEL="${ESM_MODEL:-facebook/esm2_t6_8M_UR50D}"
ESM_POOLING="${ESM_POOLING:-mean}"
ESM_PROJECT_DIM="${ESM_PROJECT_DIM:-80}"
ESM_ZSCORE="${ESM_ZSCORE:-true}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.3}"
ENCODE_CHUNKS="${ENCODE_CHUNKS:-14}"
# Comma-separated list of variants to run. Empty = run all.
# Valid names: orig, full, noesm, nosym, noskip, gated
# Example: RUN_ONLY=orig bash scripts/run_component_ablation.sh 27K bfs 100
RUN_ONLY="${RUN_ONLY:-}"

should_run() { [[ -z "${RUN_ONLY}" ]] || [[ ",${RUN_ONLY}," == *",$1,"* ]]; }

DATA_FILE="data/${DATASET}.txt"
STRUCT_PREFIX="features/${DATASET}"
OUT_DIR="${OUT_DIR:-ablation_component_${DATASET}_${SPLIT}}"
ESM_CACHE="features/${DATASET}_${ESM_MODEL//\//_}_${ESM_POOLING}.pt"
SPLIT_FILE="${I3:-data/${DATASET}_${SPLIT}.json}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "split file not found: ${SPLIT_FILE}" >&2
  echo "pass a split file with I3=/path/to/split.json" >&2
  exit 1
fi

if [[ ! -f "${ESM_CACHE}" ]]; then
  echo "ESM cache not found: ${ESM_CACHE}" >&2
  echo "precompute it first or override ESM_CACHE env var" >&2
  exit 1
fi

SEQ_FILE="$(sed -n '1p' "${DATA_FILE}")"
SEEDS_NORMALIZED="${SEEDS//,/ }"
MODEL_PREFIXES=(
  "hippi_orig_baseline"
  "hippi_pair_gated"
  "hippi_riem_noskip"
  "hippi_riem_nosym"
  "hippi_riem_noesm"
  "hippi_riem_branch_gate"
)

echo "=== Component ablation: hippi_riem_branch_gate ==="
echo "dataset: ${DATASET}"
echo "split: ${SPLIT}"
echo "seeds: ${SEEDS}"
echo "epochs: ${EPOCHS}"
echo "batch size: ${BATCH_SIZE}"
echo "split file (-i3): ${SPLIT_FILE}"
echo "sequence file: ${SEQ_FILE}"
echo "structure prefix: ${STRUCT_PREFIX}"
echo "ESM model: ${ESM_MODEL}"
echo "ESM pooling: ${ESM_POOLING}"
echo "ESM cache: ${ESM_CACHE}"
echo "ESM projection dim: ${ESM_PROJECT_DIM}"
echo "ESM z-score: ${ESM_ZSCORE}"
echo "ESM projector dropout: ${FUSION_DROPOUT}"
echo "encode_chunks: ${ENCODE_CHUNKS}"
echo "output dir: ${OUT_DIR}"

for SEED_VALUE in ${SEEDS_NORMALIZED}; do

  # ------------------------------------------------------------------ #
  # Core args shared by all 5 variants (no encoder, no edge mode,       #
  # no pair head — those are set per-variant below).                    #
  # ------------------------------------------------------------------ #
  COMMON_CORE=(
    "${PYTHON}" main.py
    -m read
    -t HI-PPI
    -i "${DATA_FILE}"
    -i3 "${SPLIT_FILE}"
    -i4 "${STRUCT_PREFIX}"
    -e "${EPOCHS}"
    -b "${BATCH_SIZE}"
    -seed "${SEED_VALUE}"
    -mainfold Hyperboloid
    -riemann_detach_encoder true
    -grad_clip_norm 5.0
    -encode_chunks "${ENCODE_CHUNKS}"
  )

  # ESM2 encoder args — appended by variants that use ESM2
  ESM_ARGS=(
    -seq_encoder esm2
    -esm_project_dim "${ESM_PROJECT_DIM}"
    -esm_zscore "${ESM_ZSCORE}"
    -fusion_dropout "${FUSION_DROPOUT}"
    -esm_cache "${ESM_CACHE}"
  )

  # ------------------------------------------------------------------ #
  # [0/5] Original baseline: handcrafted + original edges + gated head
  #        Anchor for measuring the total gain of all 3 improvements.
  # ------------------------------------------------------------------ #
  ORIG_OUT="${OUT_DIR}/hippi_orig_baseline_seed${SEED_VALUE}"
  ORIG_CMD=(
    "${COMMON_CORE[@]}"
    -seq_encoder handcrafted
    -typed_edge_mode original
    -o "${ORIG_OUT}"
    -pair_head gated
  )

  # ------------------------------------------------------------------ #
  # [1/5] Full model                                                    #
  # ------------------------------------------------------------------ #
  FULL_OUT="${OUT_DIR}/hippi_riem_branch_gate_seed${SEED_VALUE}"
  FULL_CMD=(
    "${COMMON_CORE[@]}"
    "${ESM_ARGS[@]}"
    -typed_edge_mode sym
    -o "${FULL_OUT}"
    -pair_head riemann_branch_gate
    -riemann_residual false
  )

  # ------------------------------------------------------------------ #
  # [2/5] Disable ESM2: replace with handcrafted features
  # ------------------------------------------------------------------ #
  NOESM_OUT="${OUT_DIR}/hippi_riem_noesm_seed${SEED_VALUE}"
  NOESM_CMD=(
    "${COMMON_CORE[@]}"
    -seq_encoder handcrafted
    -typed_edge_mode sym
    -o "${NOESM_OUT}"
    -pair_head riemann_branch_gate
    -riemann_residual false
  )

  # ------------------------------------------------------------------ #
  # [3/5] Disable Symmetrized Adjacency: use original edges
  # ------------------------------------------------------------------ #
  NOSYM_OUT="${OUT_DIR}/hippi_riem_nosym_seed${SEED_VALUE}"
  NOSYM_CMD=(
    "${COMMON_CORE[@]}"
    "${ESM_ARGS[@]}"
    -typed_edge_mode original
    -o "${NOSYM_OUT}"
    -pair_head riemann_branch_gate
    -riemann_residual false
  )

  # ------------------------------------------------------------------ #
  # [4/5] Disable Residual Skip: pure Riemann head, no x_v1_pair
  # ------------------------------------------------------------------ #
  NOSKIP_OUT="${OUT_DIR}/hippi_riem_noskip_seed${SEED_VALUE}"
  NOSKIP_CMD=(
    "${COMMON_CORE[@]}"
    "${ESM_ARGS[@]}"
    -typed_edge_mode sym
    -o "${NOSKIP_OUT}"
    -pair_head riemann
    -riemann_residual false
  )

  # ------------------------------------------------------------------ #
  # [5/5] Disable Riemannian Pair Head: gated baseline
  # ------------------------------------------------------------------ #
  GATED_OUT="${OUT_DIR}/hippi_pair_gated_seed${SEED_VALUE}"
  GATED_CMD=(
    "${COMMON_CORE[@]}"
    "${ESM_ARGS[@]}"
    -typed_edge_mode sym
    -o "${GATED_OUT}"
    -pair_head gated
  )

  echo
  echo "=== Seed ${SEED_VALUE} ==="

  if should_run "orig"; then
    echo
    echo ">>> [0/5] Original baseline (handcrafted + original + gated)"
    echo "${ORIG_CMD[*]}"
    "${ORIG_CMD[@]}"
  fi

  if should_run "full"; then
    echo
    echo ">>> [1/5] Full model (riem_branch_gate + esm2 + sym)"
    echo "${FULL_CMD[*]}"
    "${FULL_CMD[@]}"
  fi

  if should_run "noesm"; then
    echo
    echo ">>> [2/5] Disable ESM2 (handcrafted features)"
    echo "${NOESM_CMD[*]}"
    "${NOESM_CMD[@]}"
  fi

  if should_run "nosym"; then
    echo
    echo ">>> [3/5] Disable Symmetrized Adjacency (original edges)"
    echo "${NOSYM_CMD[*]}"
    "${NOSYM_CMD[@]}"
  fi

  if should_run "noskip"; then
    echo
    echo ">>> [4/5] Disable Residual Skip (pure Riemann, no x_v1_pair)"
    echo "${NOSKIP_CMD[*]}"
    "${NOSKIP_CMD[@]}"
  fi

  if should_run "gated"; then
    echo
    echo ">>> [5/5] Disable Riemannian Head (gated Euclidean baseline)"
    echo "${GATED_CMD[*]}"
    "${GATED_CMD[@]}"
  fi

done

echo
echo ">>> Summarize multi-seed results"
echo "${PYTHON} scripts/summarize_ablation.py --out-dir ${OUT_DIR} --seeds ${SEEDS} --models ${MODEL_PREFIXES[*]}"
"${PYTHON}" scripts/summarize_ablation.py \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEEDS}" \
  --models "${MODEL_PREFIXES[*]}"
