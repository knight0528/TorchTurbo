#!/bin/bash

if ! pip show mindspeed 2>/dev/null | grep -qx 'Version: 0.15.3'; then
    rm -rf /opt/MindSpeed
    git clone --branch core_r0.15.3 \
        https://gitcode.com/Ascend/MindSpeed.git \
        /opt/MindSpeed
    pip install -e /opt/MindSpeed
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
pip install --force-reinstall --no-deps \
  "${repo_root}/docker/npu/pkg/flash_linear_attention_npu-26.6.0-910_93.aarch64-py3-none-any.whl"
pip install \
  "${repo_root}/docker/npu/pkg/flash_linear_attention-0.6.0-py3-none-any.whl"
