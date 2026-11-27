#!/usr/bin/env bash
# Shared vs split evaluation (docs/SPLIT_THEORY.md D) on the 12 MatSynth evaluation materials.
# One folder per experiment, one sub-folder per checkpoint:
#   runs/eval/deeptex_shared_vs_split/<N>k/   DeepTex arms, split arms at step N*1000 (30k / 35k / 40k)
#   runs/eval/matup_shared_vs_split/ep10/     MatUp shared / split MLP (per-material optimisation, 10 epochs)
# each with report.json, summary.md, montage/ (in git) and maps/ (local). DeepTex sheets are in
# montage/{overview,material,zoom}/, MatUp's one folder per training setting:
# montage/{width,noreg,zinit}/{overview,material,zoom}/ (montage.py EXPERIMENTS).
# Run from the repo root in Git Bash (or any Linux shell). RUN = <experiment>/<checkpoint>;
# deeptex/... and matup/... are accepted for the two experiment folders.
#
#   bash scripts/eval.sh infer   deeptex/30k [ARGS]   DeepTex arms -> <RUN>/maps/<arm>/ (deeptex container, detached):
#                                                     split arms at the checkpoint of step 30000, the shared
#                                                     renloss_21 at its own (only) checkpoint; ARGS -> infer_deeptex.py
#   bash scripts/eval.sh montage RUN [ARGS]           the RUN's arms vs GT as pictures -> <RUN>/montage/ (matup container,
#                                                     detached; DeepTex needs infer first); ARGS -> montage.py, e.g. --seeds 0,1,2
#   bash scripts/eval.sh score   RUN [ARGS]           bilinear + the RUN's arms, every metric -> <RUN>/{report.json,summary.md}
#                                                     (matup container, detached); ARGS -> evaluate.py, e.g. --no-perceptual
#   bash scripts/eval.sh split   RUN                  cut the RUN's montage sheets per seed -> <RUN>/montage/[<setting>/]by_seed/s<k>/
#                                                     (montage does this itself; for sheets made before, or pulled from git)
#   bash scripts/eval.sh status                       what is running, log tails, what each RUN has written
#   bash scripts/eval.sh stop STAGE                   stop a detached stage (infer / montage / score)
#   bash scripts/eval.sh migrate                      move local maps/ of the old runs/eval/{main,step30k,step35k}
#                                                     into the folders above (git already moved the results)
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
EXPERIMENTS="deeptex_shared_vs_split matup_shared_vs_split"   # montage.py EXPERIMENTS
mkdir -p "$LOGS" runs/cache/nv runs/cache/torch
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

usage() { sed -n '2,30p' "$0"; exit 1; }

STAGE=${1:-}
[ $# -gt 0 ] && shift

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

parse_run() {  # RUN -> RUN (full experiment name), ONLY (deeptex / matup), CKPT
  local exp=${1%%/*}
  CKPT=${1#*/}
  case "$exp" in
    deeptex|deeptex_shared_vs_split) exp=deeptex_shared_vs_split; ONLY=deeptex ;;
    matup|matup_shared_vs_split) exp=matup_shared_vs_split; ONLY=matup ;;
    *) echo "RUN must be <experiment>/<checkpoint> with experiment deeptex or matup, got '$1'" >&2; exit 1 ;;
  esac
  case "$CKPT" in
    */*|*' '*|""|"$1") echo "RUN must be <experiment>/<checkpoint>, e.g. deeptex/40k or matup/ep10, got '$1'" >&2; exit 1 ;;
  esac
  RUN=$exp/$CKPT
}

quote_args() { EXTRA=""; for a in "$@"; do EXTRA="$EXTRA '${a//\'/}'"; done; }

case "$STAGE" in
  infer|montage|score)
    [ $# -gt 0 ] && [ "${1#-}" = "$1" ] || usage
    parse_run "$1"; shift
    quote_args "$@"
    case "$STAGE" in
      infer)
        [ "$ONLY" = deeptex ] || { echo "infer is for DeepTex only (MatUp's maps come from matup_split.sh optim)" >&2; exit 1; }
        STEP=""
        case " $* " in
          *" --step "*) ;;                                     # given explicitly
          *) [[ "$CKPT" =~ ^([0-9]+)k$ ]] || { echo "DeepTex checkpoint folder must be <N>k (step N*1000), got '$CKPT'" >&2; exit 1; }
             STEP=" --step $(( ${BASH_REMATCH[1]} * 1000 ))" ;;
        esac
        detached infer deeptex "python /workspace/variants/eval/infer_deeptex.py --name $RUN$STEP$EXTRA" ;;
      montage)
        detached montage matup "python /workspace/variants/eval/montage.py --name $RUN --only $ONLY$EXTRA" ;;
      score)
        # LPIPS's AlexNet weights are downloaded once into runs/cache/torch (TORCH_HOME).
        detached score matup "export TORCH_HOME=/workspace/runs/cache/torch; \
            python /workspace/variants/eval/evaluate.py --name $RUN --only $ONLY$EXTRA" ;;
    esac ;;
  split)
    [ $# -gt 0 ] || usage
    parse_run "$1"
    [ -f runs/eval/$RUN/montage/montage.json ] || { echo "no runs/eval/$RUN/montage/montage.json - run montage first" >&2; exit 1; }
    ( cd docker && $DC run --rm -T matup python /workspace/variants/eval/split_sheets.py /workspace/runs/eval/$RUN ) ;;
  migrate)
    # old flat names -> experiment/checkpoint; only the local maps/ (git moved the rest)
    mv_maps() {  # old dir, new run dir, glob of arm folders
      local src=runs/eval/$1/maps dst=runs/eval/$2/maps a
      for a in $src/$3; do
        [ -d "$a" ] || continue
        mkdir -p "$dst"
        if [ -e "$dst/$(basename "$a")" ]; then
          echo "  keep $dst/$(basename "$a") (exists), old copy stays in $a"
        else
          mv "$a" "$dst/" && echo "  $a -> $dst/"
        fi
      done
    }
    mv_maps main deeptex_shared_vs_split/40k 'deeptex_*'
    mv_maps main matup_shared_vs_split/ep10 'matup_*'
    mv_maps main matup_shared_vs_split/ep10 'bilinear'
    mv_maps step30k deeptex_shared_vs_split/30k 'deeptex_*'
    mv_maps step35k deeptex_shared_vs_split/35k 'deeptex_*'
    for o in step30k step35k; do                  # copies of main's MatUp / bilinear maps
      rm -rf runs/eval/$o/maps/matup_* runs/eval/$o/maps/bilinear
    done
    for o in main step30k step35k; do
      [ -d runs/eval/$o ] || continue
      find runs/eval/$o -depth -type d -empty -delete
      if [ -d runs/eval/$o ]; then echo "left in runs/eval/$o (check, then delete):"; find runs/eval/$o | head -20
      else echo "runs/eval/$o: removed (empty)"; fi
    done ;;
  stop)
    docker stop "eval_${1:?stop STAGE: infer / montage / score}" ;;
  status)
    for s in infer montage score; do
      n=eval_$s
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}" "$n"
      [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-8}" | cut -c1-200
    done
    for e in $EXPERIMENTS; do
      for d in runs/eval/$e/*/; do
        [ -d "$d" ] || continue
        echo "-- $d"
        for a in "$d"maps/*/; do
          [ -d "$a" ] && echo "   maps $(basename "$a"): $(ls -d "$a"*/ 2>/dev/null | wc -l) material(s)"
        done
        for m in "$d"montage/ "$d"montage/*/; do              # montage/ itself, or a setting folder
          [ -d "${m}material" ] || [ -d "${m}overview" ] || continue
          c() { ls "$m$1"/*.png 2>/dev/null | wc -l || true; }
          echo "   ${m#"$d"}: $(c material) material, $(c zoom) zoom, $(c overview) overview"
          for k in "$m"by_seed/s*/; do
            [ -d "$k" ] && echo "   ${k#"$d"}: $(ls "$k"*/*.png 2>/dev/null | wc -l || true) sheet(s)"
          done
        done
        [ -f "$d/summary.md" ] && echo "   summary.md $(date -r "$d/summary.md" +%FT%T)"
      done
    done
    for o in main step30k step35k; do
      [ -d runs/eval/$o ] && echo "!! runs/eval/$o: old layout - bash scripts/eval.sh migrate"
    done
    for a in deeptex_split_nb23 deeptex_split_nb7 deeptex_split_nb7_b8 deeptex_split_nb23_rmgrey deeptex_split_nb7_rmgrey; do
      d=runs/deeptex/$a/checkpoints/esrgan_40k
      [ -d "$d" ] && echo "checkpoints $d: $(ls "$d" | grep -o 'ckpt-[0-9]*' | sort -u | tr '\n' ' ')"
    done
    for d in runs/matup_split/*/; do
      a=$(basename "$d")
      case "$a" in cache|probe) continue ;; esac
      echo "MatUp $a: $(ls "$d"s*/*/*/baseColor_opt.png 2>/dev/null | wc -l) run(s) with _opt maps"
    done
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader ;;
  *)
    usage ;;
esac
