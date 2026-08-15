#!/usr/bin/env bash
# Run the paper's full model: Ours (Full ML-PPI) over multiple seeds on the SAME split,
# producing .txt logs + .pt predictions for statistical testing.
#
#   Ours = esm2 + typed_edge_mode=sym + pair_head=riemann_branch_gate
#          (riemann_residual=false) + Hyperboloid + esm_zscore
#
# Usage:
#   bash scripts/run_paper_compare.sh [DATASET] [SPLIT] [EPOCHS]
#
# Example (5 seeds, 1 split):
#   SEEDS=1,5,10,42,60 bash scripts/run_paper_compare.sh 27K bfs 100
#
# All four splits:
#   for SP in bfs dfs; do for DS in 27K 148K; do
#     SEEDS=1,5,10,42,60 bash scripts/run_paper_compare.sh $DS $SP 100
#   done; done

set -euo pipefail

DATASET="${1:-27K}"
SPLIT="${2:-bfs}"
EPOCHS="${3:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEEDS="${SEEDS:-${SEED:-1,5,10,42,60}}"
PYTHON="${PYTHON:-python3}"

# --- ESM2 ---
ESM_MODEL="${ESM_MODEL:-facebook/esm2_t6_8M_UR50D}"
ESM_POOLING="${ESM_POOLING:-mean}"
ESM_PROJECT_DIM="${ESM_PROJECT_DIM:-80}"
ESM_ZSCORE="${ESM_ZSCORE:-true}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.3}"
ENCODE_CHUNKS="${ENCODE_CHUNKS:-14}"

DATA_FILE="data/${DATASET}.txt"
STRUCT_PREFIX="features/${DATASET}"
SPLIT_FILE="${I3:-data/${DATASET}_${SPLIT}.json}"
ESM_CACHE="${ESM_CACHE:-features/${DATASET}_${ESM_MODEL//\//_}_${ESM_POOLING}.pt}"
OUT_DIR="${OUT_DIR:-compare_${DATASET}_${SPLIT}}"

mkdir -p "${OUT_DIR}"

[[ -f "${DATA_FILE}" ]]  || { echo "missing data file: ${DATA_FILE}" >&2; exit 1; }
[[ -f "${SPLIT_FILE}" ]] || { echo "missing split file: ${SPLIT_FILE} (pass I3=/path/split.json)" >&2; exit 1; }
if [[ ! -f "${ESM_CACHE}" ]]; then
  echo "missing ESM cache: ${ESM_CACHE}" >&2
  echo "precompute it with: ${PYTHON} scripts/precompute_esm2.py -i <seq_file> -o ${ESM_CACHE} --model ${ESM_MODEL} --pooling ${ESM_POOLING}" >&2
  echo "(seq_file = line 1 of ${DATA_FILE})" >&2
  exit 1
fi

SEEDS_NORMALIZED="${SEEDS//,/ }"

echo "=== Running Ours (Full ML-PPI) ==="
echo "dataset=${DATASET} split=${SPLIT} epochs=${EPOCHS} seeds=${SEEDS}"
echo "split file=${SPLIT_FILE}  struct=${STRUCT_PREFIX}  out=${OUT_DIR}"
echo "esm cache=${ESM_CACHE} (zscore=${ESM_ZSCORE}, chunks=${ENCODE_CHUNKS})"

for S in ${SEEDS_NORMALIZED}; do
  echo
  echo "########## seed ${S} ##########"

  OUT="${OUT_DIR}/ours_full_${DATASET}_${SPLIT}_seed${S}"
  echo ">>> Ours (Full ML-PPI: esm2 + sym + riemann_branch_gate)"
  "${PYTHON}" main.py \
    -m read -t HI-PPI \
    -i "${DATA_FILE}" -i3 "${SPLIT_FILE}" -i4 "${STRUCT_PREFIX}" \
    -e "${EPOCHS}" -b "${BATCH_SIZE}" -seed "${S}" \
    -mainfold Hyperboloid \
    -seq_encoder esm2 -esm_cache "${ESM_CACHE}" \
    -esm_project_dim "${ESM_PROJECT_DIM}" -esm_zscore "${ESM_ZSCORE}" \
    -fusion_dropout "${FUSION_DROPOUT}" -encode_chunks "${ENCODE_CHUNKS}" \
    -typed_edge_mode sym \
    -pair_head riemann_branch_gate -riemann_residual false \
    -riemann_detach_encoder true -grad_clip_norm 5.0 \
    -o "${OUT}"
done

echo
echo "=== Done. .txt logs + .pt predictions are in ${OUT_DIR}/ ==="
