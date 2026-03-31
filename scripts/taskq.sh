#!/usr/bin/env bash
# taskq.sh - A simple task queue dispatcher using screen sessions
set -euo pipefail
shopt -s nullglob

# check dependencies
if ! command -v screen >/dev/null 2>&1; then
  printf 'error: screen command not found\n' >&2
  exit 1
fi
if ! command -v readlink >/dev/null 2>&1; then
  printf 'error: readlink command not found\n' >&2
  exit 1
fi

# Detect the real path of the script, resolving any symbolic links
resolve_self() {
  local src="${BASH_SOURCE[0]}"
  local dir

  [[ "$src" != /* ]] && src="$PWD/$src"

  while [[ -L "$src" ]]; do
    dir="$(cd -P "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done

  dir="$(cd -P "$(dirname "$src")" && pwd)"
  printf '%s/%s\n' "$dir" "$(basename "$src")"
}

# Get the current epoch time in seconds
# using EPOCHSECONDS if available for better performance
now_epoch() {
  if [[ ${EPOCHSECONDS+set} == set ]]; then
    printf '%s\n' "$EPOCHSECONDS"
  else
    date +%s
  fi
}

# Format command line arguments for logging
format_cmdline() {
  local out="" arg
  for arg in "$@"; do
    printf -v out '%s%q ' "$out" "$arg"
  done
  printf '%s\n' "${out% }"
}

SELF_PATH="$(resolve_self)"
SCRIPT_DIR="$(cd -P "$(dirname "$SELF_PATH")" && pwd)"
SCRIPT_NAME="$(basename "$SELF_PATH")"

STATE_DIR="$SCRIPT_DIR/.taskq"
QUEUE_DIR="$STATE_DIR/queue"
RUN_DIR="$STATE_DIR/running"
FAILED_DIR="$STATE_DIR/failed"

QUEUE_LOCK="$STATE_DIR/queue.lock"
DISPATCH_LOCK="$STATE_DIR/dispatch.lock"
TASK_LOCK="$STATE_DIR/task.lock"
SEQ_FILE="$STATE_DIR/seq"

SESSION_KEY="${SELF_PATH//\//_}"
SESSION_KEY="${SESSION_KEY//[^[:alnum:]_.-]/_}"
SESSION_NAME="dispatcher${SESSION_KEY}"

DEFAULT_IDLE_TIMEOUT="${IDLE_TIMEOUT:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"

if ! [[ "$DEFAULT_IDLE_TIMEOUT" =~ ^[0-9]+$ ]] || (( DEFAULT_IDLE_TIMEOUT < 1 )); then
  printf 'idle timeout must be an integer >= 1\n' >&2
  exit 2
fi

if ! [[ "$POLL_INTERVAL" =~ ^[0-9]+$ ]] || (( POLL_INTERVAL < 1 )); then
  printf 'poll interval must be an integer >= 1\n' >&2
  exit 2
fi

# Create necessary directories and files for the dispatcher state
ensure_state() {
  mkdir -p "$QUEUE_DIR" "$RUN_DIR" "$FAILED_DIR"
  [[ -f "$SEQ_FILE" ]] || printf '0\n' >"$SEQ_FILE"
}

# Check if the dispatcher screen session is running
screen_running() {
  screen -S "$SESSION_NAME" -Q select . >/dev/null 2>&1
}

# Extract the original command line from a job file for logging purposes
extract_cmdline() {
  local job_file="$1"
  local line

  while IFS= read -r line; do
    case "$line" in
      '# cmdline: '*)
        printf '%s\n' "${line#\# cmdline: }"
        return 0
        ;;
    esac
  done <"$job_file"

  while IFS= read -r line; do
    case "$line" in
      exec\ *)
        printf '%s\n' "${line#exec }"
        return 0
        ;;
    esac
  done <"$job_file"

  printf '<unknown>\n'
}

# Clean up any stale jobs left in the queue or running directories
cleanup_stale_jobs_locked() {
  exec 7>"$QUEUE_LOCK"
  flock 7

  local file cmd

  for file in "$QUEUE_DIR"/*.job; do
    [[ -e "$file" ]] || continue
    cmd="$(extract_cmdline "$file")"
    printf 'discarding stale queued job: %s\n' "$cmd" >&2
    rm -f -- "$file"
  done

  for file in "$RUN_DIR"/*.job; do
    [[ -e "$file" ]] || continue
    cmd="$(extract_cmdline "$file")"
    printf 'discarding stale running job: %s\n' "$cmd" >&2
    rm -f -- "$file"
  done

  rm -f -- "$QUEUE_DIR"/.*.tmp "$RUN_DIR"/.*.tmp

  exec 7>&-
}

# Enqueue a new job by creating a job file with a unique ID and the command to execute
enqueue_job() {
  exec 7>"$QUEUE_LOCK"
  flock 7

  local seq
  read -r seq <"$SEQ_FILE"
  seq=$((10#$seq + 1))
  printf '%d\n' "$seq" >"$SEQ_FILE"

  local job_id
  printf -v job_id '%010d' "$seq"

  local cmdline
  cmdline="$(format_cmdline "$@")"

  local tmp="$QUEUE_DIR/.${job_id}.$$.$RANDOM.tmp"
  local job="$QUEUE_DIR/${job_id}.job"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf '# cmdline: %s\n' "$cmdline"
    printf 'exec'
    local arg
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  } >"$tmp"

  chmod 700 "$tmp"
  mv "$tmp" "$job"

  exec 7>&-

  printf '%s\n' "$job_id"
}

# Check if there are any jobs currently in the queue
queue_has_jobs() {
  compgen -G "$QUEUE_DIR/*.job" >/dev/null
}

# Run a single job by moving it to the running directory, executing it, and handling success or failure
run_one_job() {
  local job="$1"
  local base job_id claimed failed_path rc

  base="$(basename "$job")"
  job_id="${base%.job}"
  claimed="$RUN_DIR/$base"
  failed_path="$FAILED_DIR/${job_id}.failed"

  if ! mv "$job" "$claimed" 2>/dev/null; then
    return 0
  fi

  exec 9>"$TASK_LOCK"
  flock 9

  set +e
  bash "$claimed"
  rc=$?
  set -e

  exec 9>&-

  if (( rc == 0 )); then
    rm -f "$claimed"
  else
    mv "$claimed" "$failed_path"
    printf 'job failed: %s (exit=%d)\n' "$job_id" "$rc" >&2
  fi

  return 0
}

# Main loop for the dispatcher that continuously checks for new jobs and executes them, with an idle timeout for automatic shutdown
dispatcher_loop() {
  local idle_timeout="$1"
  local idle_since=0
  local found
  local job
  local now

  ensure_state

  while :; do
    found=0

    for job in "$QUEUE_DIR"/*.job; do
      found=1
      idle_since=0
      run_one_job "$job"
    done

    if (( found == 1 )); then
      continue
    fi

    if (( idle_since == 0 )); then
      idle_since="$(now_epoch)"
    fi

    now="$(now_epoch)"
    if (( now - idle_since >= idle_timeout )); then
      exec 8>"$DISPATCH_LOCK"
      flock 8

      if queue_has_jobs; then
        idle_since=0
        exec 8>&-
        continue
      fi

      exec 8>&-
      exit 0
    fi

    sleep "$POLL_INTERVAL"
  done
}

submit_job() {
  local idle_timeout="$1"
  shift

  ensure_state

  exec 8>"$DISPATCH_LOCK"
  flock 8

  if ! screen_running; then
    cleanup_stale_jobs_locked
    screen -DmS "$SESSION_NAME" \
      bash "$SELF_PATH" --dispatcher --idle-timeout "$idle_timeout"
  fi

  local job_id
  job_id="$(enqueue_job "$@")"

  exec 8>&-

  printf '%s\n' "$job_id"
}

show_status() {
  ensure_state

  local queued_files=("$QUEUE_DIR"/*.job)
  local running_files=("$RUN_DIR"/*.job)
  local failed_files=("$FAILED_DIR"/*.failed)

  local queued="${#queued_files[@]}"
  local running="${#running_files[@]}"
  local failed="${#failed_files[@]}"

  if screen_running; then
    printf 'dispatcher: running (%s)\n' "$SESSION_NAME"
  else
    printf 'dispatcher: stopped\n'
  fi

  printf 'script path: %s\n' "$SELF_PATH"
  printf 'state dir: %s\n' "$STATE_DIR"
  printf 'queued jobs: %s\n' "$queued"
  printf 'running jobs: %s\n' "$running"
  printf 'failed jobs: %s\n' "$failed"
}

stop_dispatcher() {
  if screen_running; then
    screen -S "$SESSION_NAME" -X quit
    printf 'dispatcher stopped: %s\n' "$SESSION_NAME"
  else
    printf 'dispatcher is not running\n'
  fi
}

usage() {
  cat <<EOF
usage:
  $0 <command> [args...]
  $0 --status
  $0 --stop
  $0 --dispatcher --idle-timeout <seconds>

environment:
  IDLE_TIMEOUT   Dispatcher idle timeout in seconds, default 300
  POLL_INTERVAL  Polling interval in seconds, default 1

notes:
  - Symbolic links to the script are resolved.
  - Only the stale jobs in the queue or running directories will be cleaned up when a new dispatcher is started.
  - Stale jobs will not be executed, but their command lines will be printed to stderr.

examples:
  $0 python job.py
  $0 bash sync.sh
  IDLE_TIMEOUT=300 $0 python long_task.py
  $0 --status
EOF
}

main() {
  case "${1-}" in
    --dispatcher)
      shift

      local idle_timeout="$DEFAULT_IDLE_TIMEOUT"

      while [[ $# -gt 0 ]]; do
        case "$1" in
          --idle-timeout)
            idle_timeout="${2:?missing value for --idle-timeout}"
            shift 2
            ;;
          *)
            printf 'unknown dispatcher option: %s\n' "$1" >&2
            exit 2
            ;;
        esac
      done

      if ! [[ "$idle_timeout" =~ ^[0-9]+$ ]] || (( idle_timeout < 1 )); then
        printf 'idle timeout must be an integer >= 1\n' >&2
        exit 2
      fi

      dispatcher_loop "$idle_timeout"
      ;;
    --status)
      show_status
      ;;
    --stop)
      stop_dispatcher
      ;;
    --help|-h)
      usage
      ;;
    "")
      usage
      exit 2
      ;;
    *)
      local job_id
      job_id="$(submit_job "$DEFAULT_IDLE_TIMEOUT" "$@")"
      printf 'queued job: %s\n' "$job_id"
      printf 'dispatcher session: %s\n' "$SESSION_NAME"
      ;;
  esac
}

main "$@"
