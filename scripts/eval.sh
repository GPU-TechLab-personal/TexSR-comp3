#!/usr/bin/env bash
# Shared vs split evaluation (docs/SPLIT_THEORY.md D) on the 12 MatSynth evaluation materials.
# Two stages in two containers; results in runs/eval/<NAME>/ (NAME defaults to main).
# Run from the repo root in Git Bash (or any Linux shell):
#
#   bash scripts/eval.sh infer [NAME]          DeepTex arms -> runs/eval/<NAME>/maps/<arm>/   (deeptex container, detached)
#                                              every arm of variants/eval/arms.py whose checkpoint exists;
#                                              a missing checkpoint is skipped with a warning
#   bash scripts/eval.sh score [NAME] [ARGS]   bilinear + MatUp + every metric -> runs/eval/<NAME>/{report.json,summary.md}
#                                              (matup container, detached); ARGS go to evaluate.py, e.g. --no-perceptual
#   bash scripts/eval.sh status                what is running, log tails, what each stage has written
#   bash scripts/eval.sh stop STAGE            stop a detached stage (infer / score)
#
# Detached stages run in a container named eval_<stage>; closing the terminal does not stop them.
# Logs: runs/logs/eval_<stage>.log (EXIT code on the last line when it ends).
# score needs flip-evaluator in the matup image (docker/matup/requirements.txt):
#   cd docker && docker compose build matup
# Every arm is rendered with the GT height (none of the split arms predicts height).
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS" runs/cache/nv runs/cache/torch
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

STAGE=${1:-}
[ $# -gt 0 ] && shift
NAME=main                                   # optional NAME, then evaluate.py ARGS (score)
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then NAME=$1; shift; fi
case "$NAME" in
  */*|*' '*|"") echo "NAME must be a plain folder name, got '$NAME'" >&2; exit 1 ;;
esac

detached() {  # stage, service, shell command
  local stage=$1 svc=$2 cmd=$3 name=eval_$1
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then
    echo "$name is already running - bash scripts/eval.sh status" >&2; exit 1
  fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.prev_$(date +%Y%m%d_%H%M%S).log"
  local L=/workspace/$LOGS/$name
  ( cd docker && $DC run -d -T --name "$name" "$svc" bash -c \
      "{ date +%FT%T; nvidia-smi -L 2>/dev/null; $cmd; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
  echo "started $name   log: $LOGS/$name.log   status: bash scripts/eval.sh status"
}

case "$STAGE" in
  infer)
    detached infer deeptex "python /workspace/variants/eval/infer_deeptex.py --name $NAME" ;;
  score)
    EXTRA=""
    for a in "$@"; do EXTRA="$EXTRA '${a//\'/}'"; done
    # LPIPS's AlexNet weights are downloaded once into runs/cache/torch (TORCH_HOME).
    detached score matup "export TORCH_HOME=/workspace/runs/cache/torch; \
            python /workspace/variants/eval/evaluate.py --name $NAME$EXTRA" ;;
  stop)
    # `stop infer` parses "infer" as NAME above
    docker stop "eval_${NAME:?}" ;;
  status)
    for s in infer score; do
      n=eval_$s
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
      [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-8}" | cut -c1-200
    done
    for d in runs/eval/*/; do
      [ -d "$d" ] || continue
      echo "-- $d"
      for a in "$d"maps/*/; do
        [ -d "$a" ] && echo "   maps $(basename "$a"): $(ls -d "$a"*/ 2>/dev/null | wc -l) material(s)"
      done
      [ -f "$d/summary.md" ] && echo "   summary.md $(date -r "$d/summary.md" +%FT%T)"
    done
    for d in runs/deeptex/deeptex_split_nb23/checkpoints/esrgan_40k runs/deeptex/deeptex_split_nb7/checkpoints/esrgan_40k; do
      [ -d "$d" ] && echo "checkpoints $d: $(ls "$d" | grep -o 'ckpt-[0-9]*' | sort -u | tr '\n' ' ')"
    done
    for a in matup_shared matup_split; do
      [ -d runs/matup_split/$a ] && echo "MatUp $a: $(ls runs/matup_split/$a/s*/*/*/baseColor_opt.png 2>/dev/null | wc -l) run(s) with _opt maps"
    done
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader ;;
  *)
    sed -n '2,19p' "$0"; exit 1 ;;
esac
