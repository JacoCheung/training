#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
command -v sbatch >/dev/null 2>&1 || { echo "error: sbatch not found; run on prenyx login" >&2; exit 1; }

BASE_START_TS="${BASE_START_TS:-150}"
TOTAL_TRAIN_TS="${TOTAL_TRAIN_TS:-149}"
FINAL_EVAL_TS="${FINAL_EVAL_TS:-299}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_mlperf_training}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
RUN_BASE="${RUN_BASE:-/lustre/fsw/coreai_mlperf_training/users/junzhang/yambda_runs}"
BRANCH="${BRANCH:-$(git branch --show-current)}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
DLRM_DATA_PATH="${DLRM_DATA_PATH:-/lustre/share/coreai_dlalgo_ci/artifacts/dataset/yambda_5b/hstu_preprocessed_l2039/2026-06-09}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_base_54138621}"
PYTHONPATH="${PYTHONPATH:-/lustre/fsw/coreai_mlperf_training/users/junzhang/yambda_pip_deps/py312:}"
CKPT_EVERY_N_TS="${CKPT_EVERY_N_TS:-15}"
KEEP_LAST_N="${KEEP_LAST_N:-1}"
ENROOT_NAME="${ENROOT_NAME:-}"
CLEANUP_ENROOT_ON_EXIT="${CLEANUP_ENROOT_ON_EXIT:-}"
INITIAL_CKPT_PATH="${INITIAL_CKPT_PATH:-}"
INITIAL_CKPT_TS="${INITIAL_CKPT_TS:-}"
INITIAL_DEPENDENCY="${INITIAL_DEPENDENCY:-}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
STREAMING_SHUFFLE_FRACTION="${STREAMING_SHUFFLE_FRACTION:-}"
STREAMING_SHUFFLE_SEED="${STREAMING_SHUFFLE_SEED:-}"
INTER_WINDOW_SHUFFLE="${INTER_WINDOW_SHUFFLE:-}"
ALLOW_MULTI_NODE="${ALLOW_MULTI_NODE:-0}"
NODES="${NODES:-1}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4086}"
CHAINS="${CHAINS:-}"

maxseq_for() { case "$1" in 2039) echo 2048 ;; 4086) echo 4096 ;; *) return 1 ;; esac; }
cache_for() { [ "$1" = 4086 ] && echo "/lustre/fsw/coreai_mlperf_training/users/junzhang/yambda_cache/hstu_cache_L4086_2026-06-09" || true; }
ceil_div() { echo $((($1 + $2 - 1) / $2)); }
q() { printf '%q' "$1"; }
exp() { printf 'export %s=%s\n' "$1" "$(q "$2")"; }
opt_exp() { [ -n "$2" ] && exp "$1" "$2" || printf 'unset %s\n' "$1"; }

if [ -z "${MAX_SEQ_LEN:-}" ]; then
  MAX_SEQ_LEN="$(maxseq_for "$HISTORY_LENGTH")" || {
    echo "error: MAX_SEQ_LEN must be set for HISTORY_LENGTH=$HISTORY_LENGTH" >&2
    exit 1
  }
fi
if [ -z "${LOCAL_BATCH_SIZE:-}" ]; then
  case "$HISTORY_LENGTH" in
    2039) LOCAL_BATCH_SIZE=1024 ;;
    4086) LOCAL_BATCH_SIZE=512 ;;
    *) echo "error: LOCAL_BATCH_SIZE must be set for HISTORY_LENGTH=$HISTORY_LENGTH" >&2; exit 1 ;;
  esac
fi
if [ -z "${SEGMENT_TS:-}" ]; then
  case "$HISTORY_LENGTH" in
    2039) SEGMENT_TS=17 ;;
    4086) SEGMENT_TS=8 ;;
    *) echo "error: SEGMENT_TS must be set for HISTORY_LENGTH=$HISTORY_LENGTH" >&2; exit 1 ;;
  esac
fi
RUN_GROUP="${RUN_GROUP:-yambda_prenyx_1n_l${HISTORY_LENGTH}_bs${LOCAL_BATCH_SIZE}_chain_$(date +%Y%m%d_%H%M%S)}"
CHAIN_NAME="${CHAIN_NAME:-l${HISTORY_LENGTH}_bs${LOCAL_BATCH_SIZE}_${NODES}n}"
CHAINS="${CHAINS:-${CHAIN_NAME}:${NODES}:${LOCAL_BATCH_SIZE}:${HISTORY_LENGTH}:${MAX_SEQ_LEN}:${SEGMENT_TS}}"

TRAIN_END_TS="$((BASE_START_TS + TOTAL_TRAIN_TS - 1))"
[ "$FINAL_EVAL_TS" -gt "$TRAIN_END_TS" ] || {
  echo "error: FINAL_EVAL_TS=$FINAL_EVAL_TS must be greater than train end TS=$TRAIN_END_TS" >&2
  exit 1
}

CHAIN_SPECS=()
add_chain() {
  local name="$1" nodes="$2" bs="$3" hist="$4" maxseq="$5" seg="$6"
  [ -n "$maxseq" ] || maxseq="$(maxseq_for "$hist")" || {
    echo "error: max_seq_len must be set for history_length=$hist" >&2
    exit 1
  }
  if [ "$ALLOW_MULTI_NODE" != 1 ] && [ "$nodes" != 1 ]; then
    echo "error: single-node chain submit only accepts nodes=1; use submit_prenyx_yambda_multinode.sh for nodes=$nodes" >&2
    exit 1
  fi
  CHAIN_SPECS+=("$name|$nodes|$bs|$hist|$maxseq|$seg")
}

build_chain_specs() {
  local spec old_ifs="$IFS"
  IFS=,
  for spec in $CHAINS; do
    IFS=:
    read -r -a parts <<<"$spec"
    IFS="$old_ifs"
    if [ "${#parts[@]}" = 5 ]; then
      add_chain "${parts[0]}" "$NODES" "${parts[1]}" "${parts[2]}" "${parts[3]}" "${parts[4]}"
    elif [ "${#parts[@]}" = 6 ]; then
      add_chain "${parts[0]}" "${parts[1]}" "${parts[2]}" "${parts[3]}" "${parts[4]}" "${parts[5]}"
    else
      echo "error: bad CHAINS entry '$spec' (use name:nodes:batch:history:max_seq:segment_ts)" >&2
      exit 1
    fi
    IFS=,
  done
  IFS="$old_ifs"
}

write_submit_script() {
  local path="$1" chain="$2" seg_i="$3" ts0="$4" seg_len="$5" ts1="$6" nodes="$7" bs="$8" hist="$9" maxseq="${10}"
  local prev_ckpt="${11}" prev_ts="${12}" run_name="${13}" run_root="${14}" ckpt="${15}" tb="${16}" log="${17}" env_log="${18}" cache="${19}"
  local shuffle_offset="$((ts0 - BASE_START_TS))"
  {
    cat <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-job}"
if [ "$#" -gt 0 ]; then
  shift || true
fi
EOS
    printf 'REPO_ROOT=%s\ncd "$REPO_ROOT"\nSEGMENT_SCRIPT=%s\n' "$(q "$REPO_ROOT")" "$(q "$path")"
    exp SEG_INDEX "$seg_i"; exp PREV_CKPT_PATH "$prev_ckpt"; exp PREV_TS "$prev_ts"
    exp INITIAL_CKPT_PATH "$INITIAL_CKPT_PATH"; exp INITIAL_CKPT_TS "$INITIAL_CKPT_TS"
    exp RUN_NAME "$run_name"; exp RUN_BASE "$RUN_BASE"; exp RUN_ROOT "$run_root"
    exp LOG "$log"; exp ENV_LOG "$env_log"; exp CKPT_PATH "$ckpt"; exp TENSORBOARD_LOG_PATH "$tb"
    exp DLRM_DATA_PATH "$DLRM_DATA_PATH"; exp CONTAINER_IMAGE "$CONTAINER_IMAGE"; exp PYTHONPATH "$PYTHONPATH"
    exp YAMBDA_CACHE_DIR "$cache"; exp START_TS "$ts0"; exp NUM_TRAIN_TS "$seg_len"
    exp EVAL_HOLDOUT_TS "$FINAL_EVAL_TS"; exp EVAL_HOLDOUT_NUM_WINDOWS 1
    exp LOCAL_BATCH_SIZE "$bs"; exp HISTORY_LENGTH "$hist"; exp MAX_SEQ_LEN "$maxseq"
    exp CKPT_EVERY_N_TS "$CKPT_EVERY_N_TS"; exp KEEP_LAST_N "$KEEP_LAST_N"; exp GPUS_PER_NODE 8
    exp RUN_CHAIN "$chain"; exp RUN_SEGMENT "$seg_i"
    opt_exp ENROOT_NAME "$ENROOT_NAME"; opt_exp CLEANUP_ENROOT_ON_EXIT "$CLEANUP_ENROOT_ON_EXIT"
    opt_exp STREAMING_SHUFFLE_FRACTION "$STREAMING_SHUFFLE_FRACTION"; opt_exp STREAMING_SHUFFLE_SEED "$STREAMING_SHUFFLE_SEED"
    opt_exp INTER_WINDOW_SHUFFLE "$INTER_WINDOW_SHUFFLE"
    if [ -n "$INTER_WINDOW_SHUFFLE" ] && [ "$INTER_WINDOW_SHUFFLE" != 0 ]; then
      exp INTER_WINDOW_SHUFFLE_OFFSET "$shuffle_offset"; exp INTER_WINDOW_SHUFFLE_TOTAL_TS "$TOTAL_TRAIN_TS"
    else
      opt_exp INTER_WINDOW_SHUFFLE_OFFSET "${INTER_WINDOW_SHUFFLE_OFFSET:-}"; opt_exp INTER_WINDOW_SHUFFLE_TOTAL_TS "${INTER_WINDOW_SHUFFLE_TOTAL_TS:-}"
    fi
    opt_exp HISTORY_STRATEGY "${HISTORY_STRATEGY:-}"
    opt_exp SEED "${SEED:-}"
    opt_exp DENSE_LR "${DENSE_LR:-}"; opt_exp SPARSE_LR "${SPARSE_LR:-}"
    opt_exp MIN_HISTORY "${MIN_HISTORY:-}"; opt_exp HSTU_NUM_LAYERS "${HSTU_NUM_LAYERS:-}"
    opt_exp GRAD_CLIP_NORM "${GRAD_CLIP_NORM:-}"
    cat <<'EOS'

pass_envs=(
  CUDA_VISIBLE_DEVICES GPUS_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
  NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES ENROOT_NAME CLEANUP_ENROOT_ON_EXIT
  RUN_NAME RUN_BASE RUN_ROOT LOG ENV_LOG CKPT_PATH TENSORBOARD_LOG_PATH
  DLRM_DATA_PATH YAMBDA_CACHE_DIR PYTHONPATH LOCAL_BATCH_SIZE BATCH_SIZE
  NUM_WORKERS PREFETCH_FACTOR PERSISTENT_LOADER DOUBLE_BUFFER EVAL_EACH_WINDOW
  CKPT_EVERY_N_TS HISTORY_LENGTH MAX_SEQ_LEN NUM_TRAIN_TS START_TS
  EVAL_HOLDOUT_TS EVAL_HOLDOUT_NUM_WINDOWS EVAL_EVERY_N_WINDOWS
  NUM_TRAIN_BATCHES NUM_EVAL_BATCHES KEEP_LAST_N CKPT_TIME_INTERVAL_S
  IN_WINDOW_CKPT_FREQ CKPT_STEP_FREQ HSTU_HAMMER_KERNEL PYTORCH_CUDA_ALLOC_CONF
  TORCH_CUDA_ARCH_LIST TORCH_DIST_TIMEOUT_SECONDS TRAIN_LOG_CAPTURED
  STREAMING_SHUFFLE_FRACTION STREAMING_SHUFFLE_SEED
  INTER_WINDOW_SHUFFLE INTER_WINDOW_SHUFFLE_OFFSET INTER_WINDOW_SHUFFLE_TOTAL_TS
  HISTORY_STRATEGY
  SEED DENSE_LR SPARSE_LR MIN_HISTORY HSTU_NUM_LAYERS GRAD_CLIP_NORM
)

copy_ckpt() {
  rm -rf "$CKPT_PATH"; mkdir -p "$CKPT_PATH"
  if [ "$SEG_INDEX" = 0 ]; then
    [ -z "${INITIAL_CKPT_PATH:-}${INITIAL_CKPT_TS:-}" ] && return
    [ -n "${INITIAL_CKPT_PATH:-}" ] && [ -n "${INITIAL_CKPT_TS:-}" ] || { echo "INITIAL_CKPT_PATH and INITIAL_CKPT_TS must be set together" >&2; exit 43; }
    src="$INITIAL_CKPT_PATH/$INITIAL_CKPT_TS"; dst="$CKPT_PATH/$INITIAL_CKPT_TS"
  else
    src="$PREV_CKPT_PATH/$PREV_TS"; dst="$CKPT_PATH/$PREV_TS"
  fi
  [ -d "$src" ] || { echo "missing checkpoint: $src" >&2; exit 42; }
  cp -al "$src" "$dst" 2>/dev/null || cp -a "$src" "$dst"
}

image_uri() {
  local image="$1"
  local registry="${image%%/*}"
  local image_path="${image#*/}"
  printf 'docker://%s#%s\n' "$registry" "$image_path"
}

sync_repo() {
  echo "===== github branch sync ====="
  date -Is
  git_dir="$(git rev-parse --git-dir)"
  (
    flock -x 9
    echo "sync_lock=$git_dir/prenyx-chain-sync.lock"
EOS
    printf '    git fetch %s %s\n' "$(q "$GIT_REMOTE")" "$(q "$BRANCH")"
    printf '    if git show-ref --verify --quiet %s; then git checkout %s; else git checkout -b %s FETCH_HEAD; fi\n' \
      "$(q "refs/heads/$BRANCH")" "$(q "$BRANCH")" "$(q "$BRANCH")"
    cat <<'EOS'
    git merge --ff-only FETCH_HEAD
  ) 9>"$git_dir/prenyx-chain-sync.lock"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  git status --short --branch
  git status --porcelain=v1
  echo
}

run_training() {
  RUN_NAME="${RUN_NAME:-yambda_prenyx_l4086_bs1024_ckpt15_$(date +%Y%m%d_%H%M%S)}"
  RUN_BASE="${RUN_BASE:-/lustre/fsw/coreai_mlperf_training/users/junzhang/yambda_runs}"
  RUN_ROOT="${RUN_ROOT:-$RUN_BASE/$RUN_NAME}"
  LOG="${LOG:-$RUN_ROOT/train.log}"
  ENV_LOG="${ENV_LOG:-$RUN_ROOT/environment.log}"
  CKPT_PATH="${CKPT_PATH:-$RUN_ROOT/ckpts}"
  TENSORBOARD_LOG_PATH="${TENSORBOARD_LOG_PATH:-$RUN_ROOT/tb}"

  mkdir -p "$RUN_ROOT" "$CKPT_PATH" "$TENSORBOARD_LOG_PATH" "$(dirname "$LOG")" "$(dirname "$ENV_LOG")"
  if [ "${TRAIN_LOG_CAPTURED:-0}" != 1 ]; then
    exec > >(tee -a "$LOG") 2>&1
  fi

  echo "===== prenyx yambda training ====="
  date -Is
  hostname
  pwd
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "LOG=$LOG"
  echo "ENV_LOG=$ENV_LOG"
  echo "CKPT_PATH=$CKPT_PATH"
  echo "TENSORBOARD_LOG_PATH=$TENSORBOARD_LOG_PATH"
  echo

  {
    echo "===== prenyx yambda environment ====="
    date -Is
    hostname
    pwd
    echo
    echo "===== git metadata ====="
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "git_status_short_branch:"
    git status --short --branch
    echo "git_status_porcelain_v1:"
    git status --porcelain=v1
    echo
    echo "===== environment ====="
    env | sort
  } | tee "$ENV_LOG"
  echo

  echo "===== git metadata ====="
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  git status --short --branch
  git status --porcelain=v1
  echo

  echo "===== node / gpu metadata ====="
  nvidia-smi -L || true
  python3 - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.version.cuda,
        "available",
        torch.cuda.is_available(),
        "device_count",
        torch.cuda.device_count(),
    )
except Exception as exc:
    print("torch import failed", repr(exc))
PY
  echo

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1024}"
  export BATCH_SIZE="${BATCH_SIZE:-$LOCAL_BATCH_SIZE}"
  export NUM_WORKERS="${NUM_WORKERS:-4}"
  export PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
  export PERSISTENT_LOADER="${PERSISTENT_LOADER:-1}"
  export DOUBLE_BUFFER="${DOUBLE_BUFFER:-1}"
  export EVAL_EACH_WINDOW="${EVAL_EACH_WINDOW:-1}"
  export EVAL_EVERY_N_WINDOWS="${EVAL_EVERY_N_WINDOWS:-1}"
  export CKPT_EVERY_N_TS="${CKPT_EVERY_N_TS:-15}"
  export DLRM_DATA_PATH="${DLRM_DATA_PATH:-/lustre/share/coreai_dlalgo_ci/artifacts/dataset/yambda_5b/hstu_preprocessed_l2039/2026-06-09}"
  export HISTORY_LENGTH="${HISTORY_LENGTH:-4086}"
  export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
  export CKPT_PATH
  export KEEP_LAST_N="${KEEP_LAST_N:-1}"
  export RUN_NAME
  export TENSORBOARD_LOG_PATH
  export HSTU_HAMMER_KERNEL="${HSTU_HAMMER_KERNEL:-TRITON}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
  export TORCH_DIST_TIMEOUT_SECONDS="${TORCH_DIST_TIMEOUT_SECONDS:-1800}"

  echo "===== effective overrides ====="
  env | sort | grep -E '^(CUDA_VISIBLE_DEVICES|GPUS_PER_NODE|NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT|LOCAL_BATCH_SIZE|BATCH_SIZE|NUM_WORKERS|PREFETCH_FACTOR|PERSISTENT_LOADER|DOUBLE_BUFFER|EVAL_EACH_WINDOW|CKPT_EVERY_N_TS|CKPT_PATH|KEEP_LAST_N|RUN_NAME|TENSORBOARD_LOG_PATH|HSTU_HAMMER_KERNEL|PYTORCH_CUDA_ALLOC_CONF|TORCH_CUDA_ARCH_LIST|TORCH_DIST_TIMEOUT_SECONDS|STREAMING_SHUFFLE_FRACTION|STREAMING_SHUFFLE_SEED|INTER_WINDOW_SHUFFLE|INTER_WINDOW_SHUFFLE_OFFSET|INTER_WINDOW_SHUFFLE_TOTAL_TS|HISTORY_STRATEGY|DLRM_DATA_PATH|YAMBDA_CACHE_DIR|PYTHONPATH|HISTORY_LENGTH|MAX_SEQ_LEN|NUM_TRAIN_TS|START_TS|EVAL_HOLDOUT_TS|EVAL_HOLDOUT_NUM_WINDOWS|EVAL_EVERY_N_WINDOWS|NUM_TRAIN_BATCHES|NUM_EVAL_BATCHES|CKPT_TIME_INTERVAL_S|IN_WINDOW_CKPT_FREQ|CKPT_STEP_FREQ|SEED|DENSE_LR|SPARSE_LR|MIN_HISTORY|HSTU_NUM_LAYERS|GRAD_CLIP_NORM)=' || true
  echo

  python3 -m generative_recommenders.dlrm_v3.train.train_ranker \
    --dataset yambda-5b \
    --mode streaming-train-eval
}

run_node() {
  CONTAINER_IMAGE="${CONTAINER_IMAGE:-gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_base_54138621}"
  ENROOT_IMAGE_DIR="${ENROOT_IMAGE_DIR:-/lustre/fsw/coreai_mlperf_training/users/junzhang/enroot-images}"
  ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot-data-$USER}"
  ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-$USER}"
  ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/tmp/enroot-cache-$USER}"
  ENROOT_NAME="${ENROOT_NAME:-yambda-prenyx-${SLURM_JOB_ID:-manual}-${NODE_RANK:-0}}"
  SQSH="${SQSH:-$ENROOT_IMAGE_DIR/distributed-recommender_devel_base_54138621.sqsh}"

  mkdir -p "$ENROOT_IMAGE_DIR" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH"

  echo "===== enroot launch metadata ====="
  date -Is
  hostname
  pwd
  printf 'CONTAINER_IMAGE=%s\nSQSH=%s\nENROOT_NAME=%s\n' "$CONTAINER_IMAGE" "$SQSH" "$ENROOT_NAME"
  printf 'ENROOT_DATA_PATH=%s\nENROOT_CACHE_PATH=%s\n\n' "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"

  if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "$(image_uri "$CONTAINER_IMAGE")"
  fi

  export ENROOT_DATA_PATH ENROOT_RUNTIME_PATH ENROOT_CACHE_PATH
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

  cleanup_enroot() {
    if [ "${CLEANUP_ENROOT_ON_EXIT:-1}" = "1" ]; then
      enroot remove -f "$ENROOT_NAME" >/dev/null 2>&1 || true
      rm -rf -- "${ENROOT_DATA_PATH%/}/${ENROOT_NAME}" 2>/dev/null || true
    fi
  }
  trap cleanup_enroot EXIT

  if ! command -v squashfuse >/dev/null 2>&1; then
    if ! enroot list | grep -qx "$ENROOT_NAME"; then
      enroot create -n "$ENROOT_NAME" "$SQSH"
    fi
    ENROOT_TARGET="$ENROOT_NAME"
  else
    ENROOT_TARGET="$SQSH"
  fi

  local args=(start --rw -m /lustre:/lustre -m /home:/home)
  local name
  for name in "${pass_envs[@]}"; do
    args+=(-e "$name")
  done
  args+=("$ENROOT_TARGET" bash "$SEGMENT_SCRIPT" train)
  enroot "${args[@]}"
}

run_job() {
  LOG="${LOG:-${RUN_ROOT:-/tmp}/sbatch.log}"
  mkdir -p "$(dirname "$LOG")"

  {
    echo "===== prenyx yambda slurm job ====="
    date -Is
    hostname
    pwd
    printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nSLURM_NNODES=%s\n' \
      "${SLURM_JOB_ID:-}" "${SLURM_JOB_NODELIST:-}" "${SLURM_NNODES:-1}"

    MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -1)}"
    MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
    NNODES="${NNODES:-${SLURM_NNODES:-1}}"
    TRAIN_LOG_CAPTURED=1
    export MASTER_ADDR MASTER_PORT NNODES TRAIN_LOG_CAPTURED

    printf 'MASTER_ADDR=%s\nMASTER_PORT=%s\nNNODES=%s\n\n' "$MASTER_ADDR" "$MASTER_PORT" "$NNODES"

    srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --export=ALL bash -lc '
      set -euo pipefail
      export NODE_RANK="${SLURM_NODEID:-0}"
      echo "===== node launch ====="
      date -Is
      hostname
      echo "NODE_RANK=$NODE_RANK NNODES=${NNODES:-1} MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-}"
      bash "'"$SEGMENT_SCRIPT"'" node
    '
  } 2>&1 | tee -a "$LOG"
}

if [ "$MODE" = job ] || [ "$MODE" = sbatch ]; then
  sync_repo
  copy_ckpt
  mkdir -p "$RUN_ROOT" "$TENSORBOARD_LOG_PATH"
EOS
    printf '  echo "===== segment config ====="\n'
    printf '  echo %s\n' "$(q "chain=$chain segment=$seg_i nodes=$nodes train_ts=${ts0}..${ts1} num_train_ts=$seg_len eval_holdout_ts=$FINAL_EVAL_TS")"
    cat <<'EOS'
  echo "streaming_shuffle_fraction=${STREAMING_SHUFFLE_FRACTION:-<unset>} streaming_shuffle_seed=${STREAMING_SHUFFLE_SEED:-<unset>}"
  echo "inter_window_shuffle=${INTER_WINDOW_SHUFFLE:-<unset>} inter_window_shuffle_offset=${INTER_WINDOW_SHUFFLE_OFFSET:-<unset>} inter_window_shuffle_total_ts=${INTER_WINDOW_SHUFFLE_TOTAL_TS:-<unset>} seed=${SEED:-<gin-default>}"
  echo "initial_ckpt_path=$INITIAL_CKPT_PATH initial_ckpt_ts=$INITIAL_CKPT_TS"
  echo "enroot_name=${ENROOT_NAME:-<default>} cleanup_enroot_on_exit=${CLEANUP_ENROOT_ON_EXIT:-<default>}"
  echo "run_root=$RUN_ROOT"
  echo "ckpt_path=$CKPT_PATH"
fi

case "$MODE" in
  job|sbatch) run_job "$@" ;;
  node|enroot) run_node "$@" ;;
  train) run_training "$@" ;;
  *)
    echo "usage: $0 [job|node|train]" >&2
    exit 2
    ;;
esac
EOS
  } >"$path"
  chmod +x "$path"
}

submit_segment() {
  local chain="$1" seg_i="$2" ts0="$3" seg_len="$4" ts1="$5" nodes="$6" bs="$7" hist="$8" maxseq="$9" prev_ckpt="${10}" prev_ts="${11}" dep="${12}"
  local tag run_name run_root ckpt tb log env_log out script job_name job_id
  tag="$(printf '%03d' "$seg_i")"
  run_name="${RUN_GROUP}_${chain}_seg${tag}_ts${ts0}_${ts1}"
  run_root="${RUN_BASE}/${run_name}"
  ckpt="${run_root}/checkpoints"; tb="${run_root}/tensorboard"; log="${run_root}/train.log"; env_log="${run_root}/environment.log"
  out="${run_root}/slurm-%j.out"; script="${run_root}/submit.sh"
  job_name="$(echo "${ACCOUNT}-yambda.${chain}_${tag}" | tr -c '[:alnum:]_.-' '_')"
  mkdir -p "$run_root"
  write_submit_script "$script" "$chain" "$seg_i" "$ts0" "$seg_len" "$ts1" "$nodes" "$bs" "$hist" "$maxseq" \
    "$prev_ckpt" "$prev_ts" "$run_name" "$run_root" "$ckpt" "$tb" "$log" "$env_log" "$(cache_for "$hist")"

  local args=(--parsable --account="$ACCOUNT" --partition="$PARTITION" --nodes="$nodes" --ntasks-per-node=1 --exclusive --time="$TIME_LIMIT" --job-name="$job_name" --output="$out")
  [ -n "$dep" ] && args+=(--dependency="afterok:${dep}")
  if [ -n "$SBATCH_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    local extra=($SBATCH_EXTRA_ARGS)
    args+=("${extra[@]}")
  fi
  job_id="$(sbatch "${args[@]}" "$script")"; job_id="${job_id%%;*}"
  [ -n "$job_id" ] || { echo "sbatch did not return a job id for $chain segment $seg_i" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$chain" "$seg_i" "$job_id" "$nodes" "$ts0" "$ts1" "$seg_len" "$FINAL_EVAL_TS" "${dep:-none}" "$run_root" >>"$MANIFEST"
  printf '%s seg=%s job=%s nodes=%s ts=%s..%s len=%s eval=%s dependency=%s run=%s\n' "$chain" "$seg_i" "$job_id" "$nodes" "$ts0" "$ts1" "$seg_len" "$FINAL_EVAL_TS" "${dep:-none}" "$run_root" >&2
  echo "$job_id"
}

submit_chain() {
  local chain="$1" nodes="$2" bs="$3" hist="$4" maxseq="$5" seg_ts="$6"
  local jobs prev_job="$INITIAL_DEPENDENCY" prev_ckpt="" prev_ts="" i=0
  jobs="$(ceil_div "$TOTAL_TRAIN_TS" "$seg_ts")"
  printf 'chain=%s nodes=%s local_batch_size=%s history_length=%s max_seq_len=%s segment_ts=%s jobs=%s\n' "$chain" "$nodes" "$bs" "$hist" "$maxseq" "$seg_ts" "$jobs"
  [ -n "$INITIAL_DEPENDENCY" ] && printf 'chain=%s initial_dependency=%s\n' "$chain" "$INITIAL_DEPENDENCY"
  while [ "$i" -lt "$jobs" ]; do
    local ts0 remaining len ts1 job_id tag
    ts0=$((BASE_START_TS + i * seg_ts)); remaining=$((BASE_START_TS + TOTAL_TRAIN_TS - ts0)); len="$seg_ts"
    [ "$remaining" -lt "$len" ] && len="$remaining"
    ts1=$((ts0 + len - 1))
    job_id="$(submit_segment "$chain" "$i" "$ts0" "$len" "$ts1" "$nodes" "$bs" "$hist" "$maxseq" "$prev_ckpt" "$prev_ts" "$prev_job" | tail -1)"
    prev_job="$job_id"; tag="$(printf '%03d' "$i")"
    prev_ckpt="${RUN_BASE}/${RUN_GROUP}_${chain}_seg${tag}_ts${ts0}_${ts1}/checkpoints"; prev_ts="$ts1"; i=$((i + 1))
  done
  echo
}

build_chain_specs
[ "${#CHAIN_SPECS[@]}" -gt 0 ] || { echo "error: no chains selected" >&2; exit 1; }

printf 'branch=%s\ngit_remote=%s\ncommit=%s\n' "$BRANCH" "$GIT_REMOTE" "$(git rev-parse HEAD)"
git status --short --branch
printf 'run_group=%s\naccount=%s\npartition=%s\nbase_start_ts=%s\ntrain_end_ts=%s\ntotal_train_ts=%s\nfinal_eval_ts=%s\ntime_limit=%s\n' \
  "$RUN_GROUP" "$ACCOUNT" "$PARTITION" "$BASE_START_TS" "$TRAIN_END_TS" "$TOTAL_TRAIN_TS" "$FINAL_EVAL_TS" "$TIME_LIMIT"
printf 'enroot_name=%s\ncleanup_enroot_on_exit=%s\nchains=%s\n\n' "${ENROOT_NAME:-<default>}" "${CLEANUP_ENROOT_ON_EXIT:-<default>}" "$CHAINS"

MANIFEST="${RUN_BASE}/${RUN_GROUP}/jobs.tsv"
mkdir -p "$(dirname "$MANIFEST")"
printf 'chain\tsegment\tjob_id\tnodes\tstart_ts\tend_ts\tnum_train_ts\teval_holdout_ts\tdependency\trun_root\n' >"$MANIFEST"
for spec in "${CHAIN_SPECS[@]}"; do
  IFS='|' read -r chain nodes bs hist maxseq seg_ts <<<"$spec"
  submit_chain "$chain" "$nodes" "$bs" "$hist" "$maxseq" "$seg_ts"
done
echo "manifest=$MANIFEST"
