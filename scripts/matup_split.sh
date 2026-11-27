#!/usr/bin/env bash
# MatUp with its per-material MLP shared (as released) or split into three group MLPs
# (variants/matup_split/, docs/SPLIT_THEORY.md B). Run from the repo root in Git Bash:
#
#   bash scripts/matup_split.sh test      CPU checks of SplitMLP (parameter counts, channel isolation)
#   bash scripts/matup_split.sh prep      per material: LR renders + SwinIR-L x4 renders, cached once
#                                         in runs/matup_split/cache/ (detached, ~1 min per material)
#   bash scripts/matup_split.sh optim [ARMS] [SEEDS]
#                                         12 materials x arms x seeds {0,1,2} (detached, ~1-2 min per
#                                         run; rerun skips finished runs). Arms: matup_shared,
#                                         matup_split (width 128, 1.32x), matup_split_x1 (width 104,
#                                         1.00x), matup_split_x3 (width 227, 3.00x), matup_shared_noreg /
#                                         matup_split_noreg (metallic regulariser off), matup_shared_zinit /
#                                         matup_split_zinit (last layer starts at 0), matup_shared_st /
#                                         matup_split_st (straight-through clamp); default all ten
#   bash scripts/matup_split.sh probe [ARMS] [SEEDS]
#                                         the same runs again with clamp statistics per epoch
#                                         (docs/CLAMP_DEATH.md) -> runs/matup_split/probe/ (detached;
#                                         default matup_shared,matup_split); not an evaluation arm
#   bash scripts/matup_split.sh report    every probe.json -> runs/eval/matup_clamp_probe/ep10/
#                                         {report.json,summary.md} (seconds, no GPU)
#   bash scripts/matup_split.sh status    containers, log tail, runs done
#   bash scripts/matup_split.sh stop STAGE
#
# Needs the evaluation set on the data drive (bash scripts/data.sh import / evalset) and MatUp's
# SwinIR-L weight (bash scripts/setup.sh matup). Do not run alongside DeepTex training if VRAM is short.
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS"
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
[ -f docker/.env ] || { echo "docker/.env missing: cp docker/.env.example docker/.env" >&2; exit 1; }
DATA_ROOT=$(sed -n 's/^DATA_ROOT=//p' docker/.env | tail -1)
[ -n "$DATA_ROOT" ] || { echo "docker/.env has no DATA_ROOT (it was renamed from DAS_ROOT): cp docker/.env.example docker/.env" >&2; exit 1; }
EVAL=${DATA_ROOT%/}/TexSR-data/matsynth
DRIVER=/workspace/variants/matup_split/driver.py

detached() {  # stage, shell command
  local name=matup_split_$1 cmd=$2
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then
    echo "$name is already running - bash scripts/matup_split.sh status" >&2; exit 1
  fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.prev_$(date +%Y%m%d_%H%M%S).log"
  local L=/workspace/$LOGS/$name
  ( cd docker && $DC run -d -T --name "$name" matup bash -c \
      "{ date +%FT%T; nvidia-smi -L 2>/dev/null; $cmd; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
  echo "started $name   log: $LOGS/$name.log   status: bash scripts/matup_split.sh status"
}

need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

case "${1:-}" in
  test)
    ( cd docker && $DC run --rm -T matup python /workspace/variants/matup_split/test_split_mlp.py ) ;;
  prep)
    need "$EVAL" "bash scripts/data.sh import ../TexSR-comp2 (or evalset)"
    need external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth "bash scripts/setup.sh matup"
    detached prep "python $DRIVER prep" ;;
  optim)
    need "$EVAL" "bash scripts/data.sh import ../TexSR-comp2 (or evalset)"
    detached optim "python $DRIVER optim ${2:+--arms $2} ${3:+--seeds $3}" ;;
  probe)
    need "$EVAL" "bash scripts/data.sh import ../TexSR-comp2 (or evalset)"
    detached probe "python $DRIVER probe ${2:+--arms $2} ${3:+--seeds $3}" ;;
  report)
    ( cd docker && $DC run --rm -T matup python /workspace/variants/matup_split/probe_report.py ) ;;
  stop)
    docker stop "matup_split_${2:?stage: prep, optim or probe}" ;;
  status)
    for s in prep optim probe; do
      n=matup_split_$s
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
      [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-6}" | cut -c1-200
    done
    [ -d runs/matup_split/cache ] && echo "cache: $(ls runs/matup_split/cache | wc -l) materials"
    for d in runs/matup_split/*/s[0-9]/ runs/matup_split/probe/*/s[0-9]/; do
      [ -d "$d" ] || continue
      p=${d#runs/matup_split/}; p=${p%/}
      echo "${p%/s*} ${p##*/}: $(ls "$d"*/run.json 2>/dev/null | wc -l) done, $(ls "$d"*/probe.json 2>/dev/null | wc -l) probed"
    done
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader ;;
  *)
    sed -n '2,26p' "$0"; exit 1 ;;
esac
