#!/usr/bin/env bash
# DeepTextures + render loss, split into three group generators trained jointly
# (variants/deeptex/split/, docs/SPLIT_THEORY.md A2). Run from the repo root in Git Bash:
#
#   bash scripts/deeptex_split.sh test             unit checks of the split maths (CPU, ~1 min, attached)
#   bash scripts/deeptex_split.sh params           parameter counts of nb23 and nb7 (attached)
#   bash scripts/deeptex_split.sh smoke ARM        200 steps into runs/deeptex/<arm>_smoke   (detached)
#   bash scripts/deeptex_split.sh train ARM        the full 40k-step run (detached, hours; rerun resumes)
#   bash scripts/deeptex_split.sh status           containers, log tails, checkpoints, GPU
#   bash scripts/deeptex_split.sh stop STAGE ARM   stop a detached stage
#
# ARM: nb23 | nb7   (deeptex_split_nb23 / deeptex_split_nb7)
#      shared       deeptex_renloss_21 retrained here - only if comp2's checkpoint cannot be
#                   imported (bash scripts/data.sh import); same config and trainer as comp2
#
# Needs the grouped TFRecord on the data drive (bash scripts/data.sh import / grouped).
# Run one training at a time: both arms use the whole GPU.
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS" runs/cache/nv
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
[ -f docker/.env ] || { echo "docker/.env missing: cp docker/.env.example docker/.env" >&2; exit 1; }
DATA_ROOT=$(sed -n 's/^DATA_ROOT=//p' docker/.env | tail -1)
[ -n "$DATA_ROOT" ] || { echo "docker/.env has no DATA_ROOT (it was renamed from DAS_ROOT): cp docker/.env.example docker/.env" >&2; exit 1; }
VERIFY=${DATA_ROOT%/}/TexSR-data/deeptex/matsynth_grouped/verify_report.json

arm_name() {
  case "${1:-}" in
    nb23) echo deeptex_split_nb23 ;;
    nb7) echo deeptex_split_nb7 ;;
    shared) echo deeptex_renloss_21 ;;
    *) echo "ARM must be nb23, nb7 or shared" >&2; exit 1 ;;
  esac
}

train_cmd() {  # arm -> the trainer command line
  local name; name=$(arm_name "$1")
  if [ "$1" = shared ]; then
    echo "python /workspace/variants/deeptex/paper_loop/main.py --experiment render \
      --cfg_path /workspace/variants/deeptex/render_loss/deeptex_renloss_21.yaml --stage esrgan --losses upstream \
      --shuffle off --allow-dead-lr-schedule --pretrain-dir none --no-val --log-every 100 --render-monitor-every 100"
  else
    echo "python /workspace/variants/deeptex/split/train.py --cfg_path /workspace/variants/deeptex/split/$name.yaml \
      --allow-dead-lr-schedule --log-every 100 --render-monitor-every 100"
  fi
}

attached() {  # service, command...
  local svc=$1; shift
  ( cd docker && $DC run --rm -T "$svc" "$@" )
}

detached() {  # container name, shell command
  local name=$1 cmd=$2
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then
    echo "$name is already running - bash scripts/deeptex_split.sh status" >&2; exit 1
  fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.prev_$(date +%Y%m%d_%H%M%S).log"
  local L=/workspace/$LOGS/$name
  ( cd docker && $DC run -d -T --name "$name" deeptex bash -c \
      "{ date +%FT%T; nvidia-smi -L 2>/dev/null; $cmd; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
  echo "started $name   log: $LOGS/$name.log   status: bash scripts/deeptex_split.sh status"
}

need_data() { [ -f "$VERIFY" ] || { echo "missing $VERIFY - bash scripts/data.sh import ../TexSR-comp2 (or grouped)" >&2; exit 1; }; }

case "${1:-}" in
  test)
    attached deeptex python /workspace/variants/deeptex/split/test_split.py ;;
  params)
    for a in nb23 nb7; do
      attached deeptex python /workspace/variants/deeptex/split/train.py \
        --cfg_path "/workspace/variants/deeptex/split/$(arm_name $a).yaml" --allow-dead-lr-schedule --count-only
    done ;;
  smoke)
    need_data; name=$(arm_name "${2:-}")
    detached "${name}_smoke" "$(train_cmd "$2") --max-steps 200 --save-steps 100 \
      --checkpoint-root /workspace/runs/deeptex/${name}_smoke/checkpoints --log-root /workspace/runs/deeptex/${name}_smoke/logs" ;;
  train)
    need_data; name=$(arm_name "${2:-}")
    detached "${name}_train" "$(train_cmd "$2")" ;;
  stop)
    docker stop "$(arm_name "${3:-}")_${2:?stage: smoke or train}" ;;
  status)
    for a in nb23 nb7 shared; do
      for s in smoke train; do
        n=$(arm_name $a)_$s
        [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
        docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
        [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-6}" | cut -c1-220
      done
      for d in runs/deeptex/$(arm_name $a)_smoke/checkpoints/esrgan_40k runs/deeptex/$(arm_name $a)/checkpoints/esrgan_40k; do
        [ -d "$d" ] && echo "checkpoints $d: $(ls "$d" | grep -o 'ckpt-[0-9]*' | sort -u | tr '\n' ' ')"
      done
    done
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader ;;
  *)
    sed -n '2,19p' "$0"; exit 1 ;;
esac
