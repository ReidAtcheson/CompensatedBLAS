#!/usr/bin/env bash
set -euo pipefail

function usage() {
    cat <<'USAGE'
Run the BLAS xBLAT verification executables against a freshly built CompensatedBLAS
shared library by staging the build and install into a temporary workspace.

Usage: run_xblat_integration.sh [options]

Options:
  --build-type <type>     CMake build type (default: Release)
  --jobs <n>              Parallel build jobs (default: number of cores)
  --timeout <seconds>     Per-executable timeout; 0 disables (default: 300)
  --search-dir <path>     Directory to search for xBLAT executables.
                          May be repeated. Defaults to /usr/lib/x86_64-linux-gnu/blas
                          or the colon-separated XBLAT_SEARCH_DIRS environment variable.
  --library <name>        Library filename to preload (default: libCompensatedBLAS.so)
  --keep-temp             Preserve temporary workspace for debugging
  -h, --help              Show this help message

Environment overrides:
  BUILD_TYPE, XBLAT_JOBS, XBLAT_TIMEOUT, XBLAT_SEARCH_DIRS, XBLAT_LIBRARY

The script produces per-test logs under the temporary workspace (or the
preserved directory when --keep-temp is used) and exits with a non-zero status
if any xBLAT executable fails or times out.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${XBLAT_JOBS:-$(nproc 2>/dev/null || echo 1)}"
TIMEOUT="${XBLAT_TIMEOUT:-300}"
DEFAULT_SEARCH="/usr/lib/x86_64-linux-gnu/blas"
LIBRARY_NAME="${XBLAT_LIBRARY:-libCompensatedBLAS.so}"
KEEP_TEMP=false
SEARCH_DIRS=()

while (( "$#" )); do
    case "$1" in
        --build-type)
            [[ $# -ge 2 ]] || { echo "Missing value for --build-type" >&2; exit 2; }
            BUILD_TYPE="$2"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || { echo "Missing value for --jobs" >&2; exit 2; }
            JOBS="$2"
            shift 2
            ;;
        --timeout)
            [[ $# -ge 2 ]] || { echo "Missing value for --timeout" >&2; exit 2; }
            TIMEOUT="$2"
            shift 2
            ;;
        --search-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --search-dir" >&2; exit 2; }
            SEARCH_DIRS+=("$2")
            shift 2
            ;;
        --library)
            [[ $# -ge 2 ]] || { echo "Missing value for --library" >&2; exit 2; }
            LIBRARY_NAME="$2"
            shift 2
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ${#SEARCH_DIRS[@]} -eq 0 ]]; then
    if [[ -n "${XBLAT_SEARCH_DIRS:-}" ]]; then
        IFS=':' read -r -a SEARCH_DIRS <<< "${XBLAT_SEARCH_DIRS}"
    else
        SEARCH_DIRS=(${DEFAULT_SEARCH})
    fi
fi

TMP_ROOT="$(mktemp -d)"
BUILD_DIR="${TMP_ROOT}/build"
INSTALL_DIR="${TMP_ROOT}/install"
LOG_DIR="${TMP_ROOT}/logs"
EXEC_ROOT="${TMP_ROOT}/exec"

PYTHON_BIN=${PYTHON_BIN:-python3}
PROCESS_XBLAT_OUTPUT="${SCRIPT_DIR}/process_xblat_output.py"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Required interpreter '$PYTHON_BIN' not found in PATH" >&2
    exit 3
fi

if [[ ! -x "$PROCESS_XBLAT_OUTPUT" ]]; then
    echo "Expected helper script '$PROCESS_XBLAT_OUTPUT' to be executable" >&2
    exit 3
fi

function cleanup() {
    if [[ "$KEEP_TEMP" != true ]]; then
        rm -rf "$TMP_ROOT"
    else
        echo "Temporary files preserved at: $TMP_ROOT"
    fi
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR" "$INSTALL_DIR" "$LOG_DIR" "$EXEC_ROOT"

echo "[xBLAT] Configuring project (build type: ${BUILD_TYPE})"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

echo "[xBLAT] Building with ${JOBS} job(s)"
cmake --build "$BUILD_DIR" --parallel "${JOBS}"

echo "[xBLAT] Installing into ${INSTALL_DIR}"
cmake --install "$BUILD_DIR"

LIB_PATH="$(find "$INSTALL_DIR" -name "${LIBRARY_NAME}" -print -quit)"
if [[ -z "$LIB_PATH" ]]; then
    echo "Failed to locate ${LIBRARY_NAME} under ${INSTALL_DIR}" >&2
    exit 1
fi
LIB_DIR="$(dirname "$LIB_PATH")"

declare -a EXECUTABLES=()
for dir in "${SEARCH_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        while IFS= read -r -d '' exe; do
            EXECUTABLES+=("$exe")
        done < <(find "$dir" -maxdepth 1 -type f -name 'xblat*' -executable -print0)
    fi
done

if [[ ${#EXECUTABLES[@]} -eq 0 ]]; then
    echo "No xBLAT executables found in: ${SEARCH_DIRS[*]}" >&2
    exit 125
fi

IFS=$'\n' EXECUTABLES=($(printf '%s\n' "${EXECUTABLES[@]}" | sort))
unset IFS

command -v timeout >/dev/null 2>&1 || TIMEOUT=0

PASSED=()
FAILED=()
TIMED_OUT=()

for exe in "${EXECUTABLES[@]}"; do
    base="$(basename "$exe")"
    log_file="${LOG_DIR}/${base}.log"
    run_dir="${EXEC_ROOT}/${base}"
    raw_log_file="${run_dir}/raw.log"
    normalized_log_file="${run_dir}/${base}.txt"
    summary_file="${run_dir}/${base}_summary.tsv"
    mkdir -p "$run_dir"
    echo "[xBLAT] Running ${exe}" | tee "$log_file"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        env_vars=("LD_PRELOAD=${LIB_PATH}" "LD_LIBRARY_PATH=${LIB_DIR}:${LD_LIBRARY_PATH}")
    else
        env_vars=("LD_PRELOAD=${LIB_PATH}" "LD_LIBRARY_PATH=${LIB_DIR}")
    fi
    runner=(env "${env_vars[@]}" "$exe")

    # Locate optional stdin data file for level 2/3 tests.
    input_file=""
    if [[ "$base" =~ ^xblat([23])([sdcz])$ ]]; then
        level="${BASH_REMATCH[1]}"
        kind="${BASH_REMATCH[2]}"
        candidate="$(dirname "$exe")/${kind}blat${level}.in"
        [[ -f "$candidate" ]] && input_file="$candidate"
    elif [[ "$base" =~ ^x([sdcz])cblat([23])$ ]]; then
        kind="${BASH_REMATCH[1]}"
        level="${BASH_REMATCH[2]}"
        candidate="$(dirname "$exe")/${kind}in${level}"
        [[ -f "$candidate" ]] && input_file="$candidate"
    fi

    if [[ "$TIMEOUT" != 0 ]]; then
        runner=(timeout "${TIMEOUT}" "${runner[@]}")
    fi

    set +e
    if [[ -n "$input_file" ]]; then
        (
            cd "$run_dir"
            "${runner[@]}" <"$input_file" >"$raw_log_file" 2>&1
        )
    else
        (
            cd "$run_dir"
            "${runner[@]}" >"$raw_log_file" 2>&1
        )
    fi
    status=$?
    set -e

    if [[ -f "$raw_log_file" ]]; then
        if ! "$PYTHON_BIN" "$PROCESS_XBLAT_OUTPUT" --input "$raw_log_file" --normalized "$normalized_log_file" --summary "$summary_file"; then
            echo "[xBLAT] Warning: failed to post-process log for ${base}" | tee -a "$log_file"
        fi
    fi

    pass_lines=()
    fail_lines=()
    warn_lines=()
    if [[ -f "$summary_file" ]]; then
        while IFS=$'\t' read -r kind message; do
            case "$kind" in
                PASS) pass_lines+=("$message") ;;
                FAIL) fail_lines+=("$message") ;;
                WARN) warn_lines+=("$message") ;;
            esac
        done <"$summary_file"
    fi

    failure_note=""
    if (( ${#fail_lines[@]} )); then
        for line in "${fail_lines[@]}"; do
            failure_note+="$line"$'\n'
        done
    fi
    while IFS= read -r out_file; do
        if grep -Eiq 'FATAL[[:space:]]+ERROR|FAILED|SUSPECT' "$out_file"; then
            first_match="$(grep -Ei 'FATAL[[:space:]]+ERROR|FAILED|SUSPECT' "$out_file" | head -n 1)"
            failure_note+="$(basename "$out_file"): ${first_match}"$'\n'
        fi
    done < <(find "$run_dir" -maxdepth 1 -type f -name '*.out' -print)

    if [[ -z "$failure_note" ]]; then
        if grep -Eiq '(^|[^A-Z])FAIL([^A-Z]|$)' "$log_file"; then
            first_log_fail="$(grep -Ei '(^|[^A-Z])FAIL([^A-Z]|$)' "$log_file" | head -n 1)"
            failure_note+="log: ${first_log_fail}"$'\n'
        fi
    fi

    if (( ${#fail_lines[@]} )) && [[ $status -eq 0 ]]; then
        status=1
    fi

    if [[ -n "$failure_note" && $status -eq 0 ]]; then
        status=1
    fi

    if [[ "$TIMEOUT" != 0 && $status -eq 124 ]]; then
        echo "[xBLAT] ${base} -> TIMEOUT" | tee -a "$log_file"
        TIMED_OUT+=("$base")
    elif [[ $status -eq 0 ]]; then
        echo "[xBLAT] ${base} -> PASS" | tee -a "$log_file"
        if (( ${#pass_lines[@]} )); then
            {
                echo "[xBLAT] ${base} pass highlights:"
                for line in "${pass_lines[@]}"; do
                    echo "    $line"
                done
            } | tee -a "$log_file"
        fi
        if (( ${#warn_lines[@]} )); then
            {
                echo "[xBLAT] ${base} warnings:"
                for line in "${warn_lines[@]}"; do
                    echo "    $line"
                done
            } | tee -a "$log_file"
        fi
        PASSED+=("$base")
    else
        echo "[xBLAT] ${base} -> FAIL (exit ${status})" | tee -a "$log_file"
        if (( ${#fail_lines[@]} )); then
            {
                echo "[xBLAT] ${base} failing checks:"
                for line in "${fail_lines[@]}"; do
                    echo "    $line"
                done
            } | tee -a "$log_file"
        fi
        if [[ -n "$failure_note" ]]; then
            printf '%s' "$failure_note" | tee -a "$log_file"
        fi
        FAILED+=("$base")
    fi
    echo >>"$log_file"
    echo
    echo "  log: $log_file"
    echo "  raw log: $raw_log_file"
    echo "  text log: $normalized_log_file"
    echo "  summary: $summary_file"
    echo
    done

TOTAL=${#EXECUTABLES[@]}
PASS_COUNT=${#PASSED[@]}
FAIL_COUNT=${#FAILED[@]}
TIMEOUT_COUNT=${#TIMED_OUT[@]}

printf '\n=== xBLAT Summary ===\n'
printf 'Total:   %d\n' "$TOTAL"
printf 'Passed:  %d\n' "$PASS_COUNT"
printf 'Failed:  %d\n' "$FAIL_COUNT"
printf 'Timeout: %d\n' "$TIMEOUT_COUNT"

if [[ ${#PASSED[@]} -gt 0 ]]; then
    printf '\nPassing executables:\n'
    for exe in "${PASSED[@]}"; do
        printf '  %s\n' "$exe"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '\nFailing executables:\n'
    for exe in "${FAILED[@]}"; do
        printf '  %s\n' "$exe"
    done
fi

if [[ ${#TIMED_OUT[@]} -gt 0 ]]; then
    printf '\nTimed-out executables:\n'
    for exe in "${TIMED_OUT[@]}"; do
        printf '  %s\n' "$exe"
    done
fi

if [[ $FAIL_COUNT -eq 0 && $TIMEOUT_COUNT -eq 0 ]]; then
    exit 0
else
    exit 1
fi
