#!/usr/bin/env bash
# Environment setup on the lab PC (Windows + Docker Desktop, run in Git Bash; Linux works too),
# for everything or for one part. Every step is idempotent: if anything fails or the connection
# drops, run the same command again and it continues where it stopped.
#
#   bash scripts/setup.sh [PART ...] 2>&1 | tee -a runs/logs/setup.log
#
#   PART     what it sets up                                         needed by
#   data     dataprep image (CPU only)                               scripts/data.sh
#   deeptex  deeptex image (TF 2.10) + a GPU op. The first start     scripts/deeptex_split.sh,
#            JIT-compiles TF for sm_120 (~7-10 min, once; cached     scripts/eval.sh infer
#            in runs/cache/nv)
#   matup    MatUp's SwinIR-L x4 GAN weight + matup image            scripts/matup_split.sh,
#            (torch 2.7.1, lpips, flip) + a GPU op                   scripts/eval.sh score
#   (none) = all three, in the order above.
#
# Every PART first runs the common part: GPU + driver >= 570, docker, compose, docker/.env,
# MatSynth file count, and the submodules at their pinned commits (recursively).
# With several PARTs, a failed one does not stop the others; the summary at the end lists
# each PART as OK or FAILED, and the exit code is non-zero if any failed.
set -uo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p runs/logs runs/cache/nv
ALL="data deeptex matup"
step() { echo; echo "=== [$(date +%T)] $*"; }
die()  { echo "SETUP FAILED: $*" >&2; exit 1; }
if docker compose version >/dev/null 2>&1; then DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then DC="docker-compose"
else DC=""; fi

# fetch URL DEST SIZE  (resumable, exact size)
fetch() {
  local url=$1 dst=$2 size=$3
  if [ -f "$dst" ] && [ "$(wc -c < "$dst" | tr -d ' ')" = "$size" ]; then echo "ok   $dst"; return 0; fi
  echo "get  $dst"
  mkdir -p "$(dirname "$dst")"
  for try in 1 2 3 4 5; do
    curl -fL --retry 5 --retry-delay 10 -C - -o "$dst.partial" "$url" && break
    echo "  retry $try"; sleep 10
  done
  [ "$(wc -c < "$dst.partial" | tr -d ' ')" = "$size" ] || die "$dst: size mismatch (rerun to resume)"
  mv "$dst.partial" "$dst"
}

# ------------------------------------------------------------------ common
common() {
  step "common: preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no NVIDIA driver)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "$drv" -ge 570 ] || die "driver $drv < 570: CUDA 12.8 needs >= 570"
  command -v docker >/dev/null || die "docker not installed"
  docker info >/dev/null 2>&1 || die "docker is not running (start Docker Desktop) or needs sudo"
  [ -n "$DC" ] || die "neither 'docker compose' nor 'docker-compose' is available"
  echo "compose: $($DC version | head -1)"
  [ -f docker/.env ] || die "docker/.env missing: cp docker/.env.example docker/.env and set DAS_ROOT"
  local das n; das=$(sed -n 's/^DAS_ROOT=//p' docker/.env | tail -1)
  n=$(ls "$das"/MatSynth/data/*.parquet 2>/dev/null | wc -l)
  echo "DAS_ROOT=$das   MatSynth parquet files: $n (431 train + 8 test when complete)"
  echo "data root $das/TexSR-data: $(ls "$das"/TexSR-data 2>/dev/null | tr '
' ' ')(bash scripts/data.sh status)"

  step "common: submodules"
  git submodule update --init --recursive || die "submodule update failed (rerun to resume)"
  git submodule status
}

# ------------------------------------------------------------------ data
part_data() {
  step "data: dataprep image"
  ( cd docker && $DC build dataprep ) || die "dataprep image build failed (rerun: the layer cache resumes it)"
  ( cd docker && $DC run --rm -T dataprep python -c "import pyarrow, cv2, numpy; print('dataprep OK | pyarrow', pyarrow.__version__, '| cv2', cv2.__version__)" ) \
    || die "dataprep image does not start"
}

# ------------------------------------------------------------------ deeptex
part_deeptex() {
  step "deeptex: image"
  ( cd docker && $DC build deeptex ) || die "deeptex image build failed (rerun: the layer cache resumes it)"
  step "deeptex: GPU (first start JIT-compiles TF for this GPU - several minutes, once)"
  ( cd docker && $DC run --rm -T deeptex python /workspace/scripts/tf_gpu_test.py ) || die "deeptex image cannot use the GPU"
}

# ------------------------------------------------------------------ matup
part_matup() {
  step "matup: weights"
  [ -f external/matup_egsr/sr_modules/SwinIR/main_test_swinir.py ] || die "MatUp's SwinIR submodule missing"
  # SwinIR-L x4 GAN (external/matup_egsr/pretrained_models/download_links.txt)
  fetch "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" \
        external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth 142473939
  step "matup: image"
  ( cd docker && $DC build matup ) || die "matup image build failed (rerun: the layer cache resumes it)"
  step "matup: GPU"
  ( cd docker && $DC run --rm -T matup python -c "
import torch; x = torch.randn(4, 3, 64, 64, device='cuda'); c = torch.nn.Conv2d(3, 8, 3).cuda()
y = c(x); y.mean().backward(); print('matup GPU OK |', torch.cuda.get_device_name(), torch.cuda.get_arch_list())" ) \
    || die "matup image cannot use the GPU"
}

# ------------------------------------------------------------------ main
PARTS=${*:-$ALL}
for p in $PARTS; do
  case " $ALL " in *" $p "*) ;; *) sed -n '2,18p' "$0"; echo "unknown PART: $p" >&2; exit 1 ;; esac
done
( common ) || exit 1
summary=() rc=0
for p in $PARTS; do
  if ( "part_$p" ); then summary+=("OK      $p"); else summary+=("FAILED  $p"); rc=1; fi
done
step "summary"
printf '%s\n' "${summary[@]}"
[ $rc = 0 ] && echo "SETUP DONE: $PARTS" || echo "SETUP INCOMPLETE - rerun the FAILED part(s): bash scripts/setup.sh <PART>"
exit $rc
