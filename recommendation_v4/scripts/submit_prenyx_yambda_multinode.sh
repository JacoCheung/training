#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NODES="${NODES:-8}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4086}"
SEGMENT_TS="${SEGMENT_TS:-30}"

case "$HISTORY_LENGTH" in
  2039)
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
    LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1024}"
    ;;
  4086)
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
    LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-512}"
    ;;
  *)
    : "${MAX_SEQ_LEN:?MAX_SEQ_LEN must be set for HISTORY_LENGTH=$HISTORY_LENGTH}"
    : "${LOCAL_BATCH_SIZE:?LOCAL_BATCH_SIZE must be set for HISTORY_LENGTH=$HISTORY_LENGTH}"
    ;;
esac

RUN_BASE="${RUN_BASE:-/lustre/fsw/coreai_mlperf_training/users/junzhang/yambda_runs}"
RUN_GROUP="${RUN_GROUP:-yambda_prenyx_multinode_l${HISTORY_LENGTH}_bs${LOCAL_BATCH_SIZE}_${NODES}n_$(date +%Y%m%d_%H%M%S)}"
CHAIN_NAME="${CHAIN_NAME:-l${HISTORY_LENGTH}_bs${LOCAL_BATCH_SIZE}_${NODES}n}"
CHAINS="${CHAINS:-${CHAIN_NAME}:${NODES}:${LOCAL_BATCH_SIZE}:${HISTORY_LENGTH}:${MAX_SEQ_LEN}:${SEGMENT_TS}}"

ENROOT_NAME="${ENROOT_NAME:-yambda-prenyx-distributed-recommender_devel_base_54138621}"
CLEANUP_ENROOT_ON_EXIT="${CLEANUP_ENROOT_ON_EXIT:-0}"
ALLOW_MULTI_NODE="${ALLOW_MULTI_NODE:-1}"

export RUN_BASE RUN_GROUP CHAINS ENROOT_NAME CLEANUP_ENROOT_ON_EXIT ALLOW_MULTI_NODE
export NODES HISTORY_LENGTH MAX_SEQ_LEN LOCAL_BATCH_SIZE SEGMENT_TS

echo "===== prenyx yambda multinode submit ====="
printf 'chains=%s\nrun_group=%s\nenroot_name=%s\ncleanup_enroot_on_exit=%s\nallow_multi_node=%s\n\n' \
  "$CHAINS" "$RUN_GROUP" "$ENROOT_NAME" "$CLEANUP_ENROOT_ON_EXIT" "$ALLOW_MULTI_NODE"

exec "$REPO_ROOT/scripts/submit_prenyx_yambda_chains.sh" "$@"
