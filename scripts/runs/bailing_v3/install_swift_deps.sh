#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/../deps/swift_deps.sh"

pip install --force-reinstall --no-deps \
  "${repo_root}/docker/npu/pkg/flash_linear_attention_npu-26.6.0-910_93.aarch64-py3-none-any.whl"
pip install \
  "${repo_root}/docker/npu/pkg/flash_linear_attention-0.6.0-py3-none-any.whl"

rm -rf ~/.cache/ ~/.triton/
