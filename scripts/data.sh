#!/usr/bin/env bash
# The data every arm shares, on the data drive (DATA_ROOT in docker/.env, D:/ on the lab PC):
#
#   <DATA_ROOT>/TexSR-data/                          (/data/TexSR-data in the containers)
#     matsynth/                     evaluation set: 12 materials, LR input + 512 GT
#     datasets/matsynth/            512/128 training pairs <map>/{HR,LRx4}/, _meta/, pairs_manifest.json
#     deeptex/matsynth_grouped/     grouped TFRecord (one material's 5 maps per Example) + verify_report.json
#     checkpoints/deeptex_renloss_21/esrgan_40k/    the shared baseline's checkpoints (from comp2)
#
# Run from the repo root in Git Bash (or any Linux shell):
#
#   bash scripts/data.sh import COMP2   copy all of the above from a TexSR-comp2 checkout's runs/
#                                       (robocopy, resumable: rerun to continue)          <- usual path
#   bash scripts/data.sh verify         grouped TFRecord byte check + pair/eval degradation check
#   --- or build from MatSynth (comp2 did exactly this; same code, same output):
#   bash scripts/data.sh evalset        12 evaluation materials from test-00000 (~1-2 min)
#   bash scripts/data.sh check          pair degradation == evaluation degradation, pixel for pixel ([PASS])
#   bash scripts/data.sh pairs          all train shards -> pairs (detached, ~1 h from an HDD; resumes)
#   bash scripts/data.sh grouped        pairs -> grouped TFRecord + byte verify (detached)
#   bash scripts/data.sh status         what is on the data drive, running containers, log tails
#   bash scripts/data.sh stop STAGE     stop pairs / grouped
set -euo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGS=runs/logs
mkdir -p "$LOGS"
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
[ -f docker/.env ] || { echo "docker/.env missing: cp docker/.env.example docker/.env and set DATA_ROOT" >&2; exit 1; }
DATA_ROOT=$(sed -n 's/^DATA_ROOT=//p' docker/.env | tail -1)
[ -n "$DATA_ROOT" ] || { echo "docker/.env has no DATA_ROOT (it was renamed from DAS_ROOT): cp docker/.env.example docker/.env" >&2; exit 1; }
HOST=${DATA_ROOT%/}/TexSR-data              # host side
DATA=/data/TexSR-data                 # container side
GROUPED=$DATA/deeptex/matsynth_grouped/matsynth_grouped_lrx4.tfrecord

need() { [ -e "$1" ] || { echo "missing $1 - $2" >&2; exit 1; }; }

detached() {  # name, service, shell command
  local name=$1 svc=$2 cmd=$3
  if [ -n "$(docker ps -q -f "name=^${name}$")" ]; then echo "$name is already running - bash scripts/data.sh status" >&2; exit 1; fi
  [ -n "$(docker ps -aq -f "name=^${name}$")" ] && docker rm "$name" >/dev/null
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.prev_$(date +%Y%m%d_%H%M%S).log"
  local L=/workspace/$LOGS/$name
  ( cd docker && $DC run -d -T --name "$name" "$svc" bash -c \
      "{ date +%FT%T; $cmd; } > $L.log 2>&1; echo \"EXIT \$?\" >> $L.log" )
  echo "started $name   log: $LOGS/$name.log   status: bash scripts/data.sh status"
}

copy() {  # src dst - robocopy on Windows (restartable, multithreaded), cp elsewhere
  local src=$1 dst=$2
  if [ ! -e "$src" ]; then echo "skip  $src (not there)"; return 0; fi
  echo "copy  $src -> $dst"
  mkdir -p "$dst"
  if command -v robocopy >/dev/null 2>&1; then
    local rc=0
    robocopy "$(cygpath -w "$src")" "$(cygpath -w "$dst")" /E /Z /R:3 /W:5 /MT:8 /NFL /NDL /NP /NJH || rc=$?
    [ "$rc" -lt 8 ] || { echo "robocopy failed ($rc) on $src - rerun to resume" >&2; exit 1; }
  else
    cp -rn "$src"/. "$dst"/
  fi
}

case "${1:-}" in
  import)
    C2=${2:?path of the TexSR-comp2 checkout, e.g. ../TexSR-comp2}
    need "$C2/runs" "is $C2 a TexSR-comp2 checkout with its runs/?"
    copy "$C2/runs/matsynth"                                     "$HOST/matsynth"
    copy "$C2/runs/datasets/matsynth"                            "$HOST/datasets/matsynth"
    copy "$C2/runs/deeptex/datasets/matsynth_grouped"            "$HOST/deeptex/matsynth_grouped"
    copy "$C2/runs/deeptex/deeptex_renloss_21/checkpoints/esrgan_40k" "$HOST/checkpoints/deeptex_renloss_21/esrgan_40k"
    echo "import done -> $HOST. Next: bash scripts/data.sh verify" ;;
  verify)
    need "$HOST/matsynth" "bash scripts/data.sh import ../TexSR-comp2 (or evalset)"
    need "$HOST/deeptex/matsynth_grouped/verify_report.json" "bash scripts/data.sh import ../TexSR-comp2 (or grouped)"
    ( cd docker && $DC run --rm -T dataprep python variants/deeptex/matsynth/matsynth_make_pairs.py --check-eval )
    # A copied record's meta still names the pair folder it was built from; point it at the
    # copy (old value kept). The verifier below then compares every PNG string against it.
    ( cd docker && $DC run --rm -T dataprep python variants/util/rebase_grouped_root.py \
        --tfrecord $GROUPED --root $DATA/datasets/matsynth )
    ( cd docker && $DC run --rm -T deeptex python /workspace/variants/deeptex/render_loss/build/texsr_verify_grouped.py \
        --tfrecord $GROUPED --root $DATA/datasets/matsynth ) ;;
  evalset)
    ( cd docker && $DC run --rm -T dataprep python variants/util/matsynth_extract.py --index 0 --count 12 --out $DATA/matsynth ) ;;
  check)
    need "$HOST/matsynth" "run: bash scripts/data.sh evalset"
    ( cd docker && $DC run --rm -T dataprep python variants/deeptex/matsynth/matsynth_make_pairs.py --check-eval ) ;;
  pairs)
    need "$HOST/matsynth" "run: bash scripts/data.sh evalset"
    detached data_pairs dataprep "python variants/deeptex/matsynth/matsynth_make_pairs.py --out $DATA/datasets/matsynth --jobs 4" ;;
  grouped)
    need "$HOST/datasets/matsynth/pairs_manifest.json" "run: bash scripts/data.sh pairs (and let it finish)"
    detached data_grouped deeptex "python /workspace/variants/deeptex/render_loss/build/texsr_build_grouped_tfrecord.py \
        --root $DATA/datasets/matsynth --out $GROUPED && python /workspace/variants/deeptex/render_loss/build/texsr_verify_grouped.py \
        --tfrecord $GROUPED --root $DATA/datasets/matsynth" ;;
  stop)
    docker stop "data_${2:?pairs or grouped}" ;;
  status)
    echo "data root: $HOST"
    [ -d "$HOST/matsynth" ] && echo "evalset: $(ls "$HOST/matsynth" | wc -l) materials" || echo "evalset: missing"
    [ -d "$HOST/datasets/matsynth/_meta" ] && echo "pairs: $(ls "$HOST/datasets/matsynth/_meta" | wc -l) materials" || echo "pairs: missing"
    [ -f "$HOST/datasets/matsynth/pairs_manifest.json" ] && echo "pairs_manifest.json present (a pairs run finished)"
    [ -f "$HOST/deeptex/matsynth_grouped/verify_report.json" ] && echo "grouped: $(grep -o '"verdict": *"[A-Z]*"' "$HOST/deeptex/matsynth_grouped/verify_report.json")" || echo "grouped: missing"
    d="$HOST/checkpoints/deeptex_renloss_21/esrgan_40k"
    [ -d "$d" ] && echo "deeptex_renloss_21 checkpoints: $(ls "$d" | grep -o 'ckpt-[0-9]*' | sort -u | tr '\n' ' ')" || echo "deeptex_renloss_21 checkpoints: missing"
    for n in data_pairs data_grouped; do
      [ -n "$(docker ps -aq -f "name=^${n}$")" ] || continue
      docker inspect -f "== $n: {{.State.Status}} exit={{.State.ExitCode}} started={{.State.StartedAt}}" "$n"
      [ -f "$LOGS/$n.log" ] && tr '\r' '\n' < "$LOGS/$n.log" | grep -v '^\s*$' | tail -"${LINES:-6}" | cut -c1-200
    done
    true ;;
  *)
    sed -n '2,23p' "$0"; exit 1 ;;
esac
