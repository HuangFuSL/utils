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
  screen -ls 2>/dev/null | grep -Fq ".${SESSION_NAME}"
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

  local submit_cwd="$PWD"
  local -a argv=("$@")
  local target_cmd="${argv[0]}"

  if [[ "$target_cmd" != */* ]]; then
    if [[ -x "$submit_cwd/$target_cmd" ]]; then
      argv[0]="./$target_cmd"
    else
      if ! local command_path="$(command -v "$target_cmd" 2>/dev/null)"; then
        printf 'error: command not found: %s\n' "$target_cmd" >&2
        exit 127
      fi
      argv[0]="$command_path"
    fi
  fi

  local cmdline
  cmdline="$(format_cmdline "${argv[@]}")"

  local tmp="$QUEUE_DIR/.${job_id}.$$.$RANDOM.tmp"
  local job="$QUEUE_DIR/${job_id}.job"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf '# cwd: %s\n' "$submit_cwd"
    printf '# cmdline: %s\n' "$cmdline"
    printf 'export PATH=%q\n' "$PATH"
    printf 'cd %q\n' "$submit_cwd"
    printf 'exec'
    local arg
    for arg in "${argv[@]}"; do
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
    (
      exec 7>&- 8>&- 9>&-
      exec </dev/null >/dev/null 2>&1
      exec screen -dmS "$SESSION_NAME" \
        bash "$SELF_PATH" --dispatcher --idle-timeout "$idle_timeout"
    )
  fi

  SUBMITTED_JOB_ID="$(enqueue_job "$@")"

  exec 8>&-
}

show_status() {
  ensure_state

  local is_running=false
  if screen_running; then
    is_running=true
  fi

  local queued_files=("$QUEUE_DIR"/*.job)
  local running_files=("$RUN_DIR"/*.job)
  local failed_files=("$FAILED_DIR"/*.failed)

  local queued="${#queued_files[@]}"
  local failed="${#failed_files[@]}"

  local bold_green=$'\033[1;32m'
  local bold_red=$'\033[1;31m'
  local bold_yellow=$'\033[1;33m'
  local gray=$'\033[90m'
  local reset=$'\033[0m'

  if $is_running; then
    printf 'Dispatcher: %sRUNNING%s at (%s)\n' "$bold_green" "$reset" "$SESSION_NAME"
  else
    printf 'Dispatcher: %sSTOPPED%s\n' "$bold_red" "$reset"
  fi

  printf '%sScript path: %s%s\n' "$gray" "$SELF_PATH" "$reset"
  printf '%sState dir: %s%s\n' "$gray" "$STATE_DIR" "$reset"

  if $is_running; then
    if (( ${#running_files[@]} > 0 )); then
      local current_cmdline
      current_cmdline="$(extract_cmdline "${running_files[0]}")"
      printf '%sCurrent job%s: %s\n' "$bold_green" "$reset" "$current_cmdline"
    else
      printf '%sNo running jobs%s\n' "$bold_yellow" "$reset"
    fi

    if (( queued > 0 )); then
      local upcoming_cmdline
      upcoming_cmdline="$(extract_cmdline "${queued_files[0]}")"
      printf '%sUpcoming job%s: %s\n' "$bold_yellow" "$reset" "$upcoming_cmdline"
    fi

    printf '%sQueued jobs: %s%s\n' "$bold_yellow" "$queued" "$reset"
    printf '%sFailed jobs: %s%s\n' "$bold_red" "$failed" "$reset"
  fi
}

handle_kill() {
  ensure_state

  local -a queued_files=("$QUEUE_DIR"/*.job)
  local -a running_files=("$RUN_DIR"/*.job)
  local queued="${#queued_files[@]}"
  local running="${#running_files[@]}"
  local reply
  local job

  local bold_red=$'\033[1;31m'
  local reset=$'\033[0m'

  if ! screen_running; then
    printf 'Dispatcher is not running; nothing to do.\n'
    return 0
  fi

  printf 'Trying to stop dispatcher session: %s\n' "$SESSION_NAME"

  if (( running == 0 )); then
    stop_dispatcher
    return 0
  fi

  printf '%sWARNING: dispatcher is currently running task(s). Killing it will interrupt them immediately.%s\n' "$bold_red" "$reset"
  printf '%sInterrupted or queued tasks will not be resumed automatically after termination.%s\n' "$bold_red" "$reset"
  printf 'Running task(s):\n'
  for job in "${running_files[@]}"; do
    printf '  - %s\n' "$(extract_cmdline "$job")"
  done
  printf 'Queued jobs: %d\n' "$queued"

  printf 'Terminate dispatcher? [y/N] '
  if ! read -r reply; then
    printf '\nKill cancelled\n'
    return 1
  fi

  case "$reply" in
    [Yy]|[Yy][Ee][Ss])
      stop_dispatcher
      ;;
    *)
      printf 'Kill cancelled\n'
      ;;
  esac
}

stop_dispatcher() {
  if screen_running; then
    screen -S "$SESSION_NAME" -X quit
  fi
}

usage() {
  cat <<EOF
usage:
  $0 <command> [args...]
  $0 --status
  $0 --kill
  $0 --dispatcher --idle-timeout <seconds>

environment:
  IDLE_TIMEOUT   Dispatcher idle timeout in seconds, default 300
  POLL_INTERVAL  Polling interval in seconds, default 1

notes:
  - Symbolic links to the script are resolved.
  - [IMPORTANT] All the stale jobs in the queue or running directories will be cleaned up when a new dispatcher is started. Stale jobs will not be executed, but their command lines will be printed to stderr.

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
    --kill)
      handle_kill
      ;;
    --help|-h)
      usage
      ;;
    "")
      usage
      exit 2
      ;;
    *)
      submit_job "$DEFAULT_IDLE_TIMEOUT" "$@"
      printf 'queued job: %s\n' "$SUBMITTED_JOB_ID"
      printf 'dispatcher session: %s\n' "$SESSION_NAME"
      ;;
  esac
}

main "$@"
