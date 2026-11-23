# MatUp - "Repurposing Image Upsamplers for SVBRDFs" (EGSR 2024)
# https://github.com/AlbanGauthier/matup_egsr @ c74ec99
#
# RTX 50-series (Blackwell, sm_120) build. Upstream pins torch 2.0.0 / CUDA 11.7,
# whose wheels stop at sm_86 with no usable PTX: on an RTX 5080 torch 2.0.0 prints
# "sm_120 is not compatible" and every kernel fails (measured 2026-11-17). torch
# 2.7.1 / CUDA 12.8 is the first release with sm_120 kernels, the same stack as the
# pbrsr image. This is a recorded deviation from upstream's pin (docs/DEVIATIONS.md).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN python -c "\
from timm.models.layers import DropPath, to_2tuple, trunc_normal_; \
import requests, cv2, lpips, torch, torchvision; \
print('matup deps OK | torch', torch.__version__, '| torchvision', torchvision.__version__, '| arch', torch.cuda.get_arch_list())"

WORKDIR /workspace/external/matup_egsr
CMD ["bash"]
