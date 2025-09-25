#!/usr/bin/env bash

set -euo pipefail

# Run the CMake configure/build/test flow inside an isolated temporary directory.

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cmake -S "${PWD}" -B "${tmp_dir}"
cmake --build "${tmp_dir}"
ctest --test-dir "${tmp_dir}" --output-on-failure
