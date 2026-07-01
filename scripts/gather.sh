#!/usr/bin/env bash
# gather.sh — parallel command executor with concurrency control
#   gather.sh [--max-concurrent N] [--fail-fast] "cmd1" "cmd2" ...
#   Each argument runs via bash -c. Exit 0 iff every command succeeds.

set -euo pipefail
set -m  # enable job control so background jobs get their own process group

MAX_CONCURRENT=0  # 0 means unlimited
FAIL_FAST=false
COMMANDS=()

tag_output() {
  local tag="$1"
  awk -v tag="$tag" '{print tag $0; fflush()}'
}

usage() {
  cat <<'EOF'
usage: gather.sh [options] "cmd1" ["cmd2" ...]

options:
  --max-concurrent N   Maximum parallel commands (default: 0 = unlimited)
  --fail-fast          Kill all remaining commands on first failure
  --help, -h           Show this message

Each positional argument is a shell command executed via bash -c.
Metadata lines are prefixed with [Instance #N]#, task output with [Instance #N] (no #).

examples:
  gather.sh "echo hello" "echo world"
  gather.sh --max-concurrent 2 "sleep 5" "sleep 3" "sleep 1"
  gather.sh --fail-fast --max-concurrent 4 "cmd1" "cmd2" "cmd3"

log grepping:
  # task output for instance N (no metadata lines)
  grep '^\[Instance #N\]\($\|[^#]\)' log

  # metadata for instance N
  grep '^\[Instance #N\]#' log

  # all command lines
  grep '^\[Instance #[0-9]\+\]# cmdline:' log

  # failed tasks
  grep '^\[Instance #[0-9]\+\]# exit_code:' log | grep -v ': 0$'

  # finish timestamps
  grep '^\[Instance #[0-9]\+\]# finished:' log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-concurrent)
      MAX_CONCURRENT="${2:?missing value for --max-concurrent}"
      if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || (( MAX_CONCURRENT < 0 )); then
        printf 'error: --max-concurrent must be a non-negative integer\n' >&2
        exit 2
      fi
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        COMMANDS+=("$1")
        shift
      done
      break
      ;;
    -*)
      printf 'error: unknown option: %s\n' "$1" >&2
      exit 2
      ;;
    *)
      COMMANDS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  printf 'error: no commands provided\n\n' >&2
  usage >&2
  exit 2
fi

declare -a PIDS=()
declare -a INSTANCES=()
FAILED=0
RUNNING=0
NEXT=0
TOTAL=${#COMMANDS[@]}

cleanup() {
  for pid in "${PIDS[@]}"; do
    [[ -z "$pid" ]] && continue
    kill -TERM -"$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

launch_one() {
  local instance=$1
  local cmd=$2
  local tag="[Instance #$instance]"
  local safe_cmdline="${cmd//$'\n'/\\n}"

  (
    printf '%s# gather job: Instance #%d\n' "$tag" "$instance"
    printf '%s# cmdline: %s\n' "$tag" "$safe_cmdline"
    printf '%s# started: %s\n' "$tag" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s#\n' "$tag"

    set +eo pipefail
    bash -c "$cmd" 2>&1 | tag_output "$tag"
    rc=${PIPESTATUS[0]}

    printf '%s# finished: %s\n' "$tag" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s# exit_code: %d\n' "$tag" "$rc"
    exit $rc
  ) &
  local pid=$!

  PIDS+=("$pid")
  INSTANCES+=("$instance")
  RUNNING=$((RUNNING + 1))
}

kill_remaining() {
  local i
  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    [[ -z "$pid" ]] && continue
    kill -KILL -- "-$pid" 2>/dev/null || true
    # wait "$pid" 2>/dev/null || true
    PIDS[$i]=""
    INSTANCES[$i]=0
  done
  RUNNING=0
}

reap_dead() {
  local has_fail=0
  local i

  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    [[ -z "$pid" ]] && continue

    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    local instance="${INSTANCES[$i]}"
    local exit_code=0

    set +e
    wait "$pid" 2>/dev/null
    exit_code=$?
    set -e

    if [[ $exit_code -ne 0 ]]; then
      printf '[Instance #%d] exited with code %d\n' "$instance" "$exit_code" >&2
      FAILED=1
      has_fail=1

      if $FAIL_FAST; then
        kill_remaining
        return 2
      fi
    fi

    PIDS[$i]=""
    INSTANCES[$i]=0
    RUNNING=$((RUNNING - 1))
  done

  return $has_fail
}

while [[ $NEXT -lt $TOTAL || $RUNNING -gt 0 ]]; do

  reap_dead || {
    if [[ $? -eq 2 ]]; then
      trap - EXIT
      exit 1
    fi
  }

  while [[ $NEXT -lt $TOTAL ]]; do
    if [[ $MAX_CONCURRENT -gt 0 && $RUNNING -ge $MAX_CONCURRENT ]]; then
      break
    fi
    launch_one "$((NEXT + 1))" "${COMMANDS[$NEXT]}"
    NEXT=$((NEXT + 1))
  done

  [[ $RUNNING -eq 0 ]] && break

  # bash >= 4.3 has wait -n; fallback to brief sleep for older versions
  if [[ ${BASH_VERSINFO[0]} -ge 5 ]] || \
     [[ ${BASH_VERSINFO[0]} -eq 4 && ${BASH_VERSINFO[1]} -ge 3 ]]; then
    set +e
    wait -n 2>/dev/null
    set -e
  else
    sleep 1
  fi

done

trap - EXIT
exit $FAILED
