#!/usr/bin/env bash
# taskq.sh - A simple task queue dispatcher (screen or nohup backend)
set -euo pipefail
shopt -s nullglob

# check dependencies
if command -v screen >/dev/null 2>&1; then
  readonly BACKEND=screen
elif command -v nohup >/dev/null 2>&1; then
  readonly BACKEND=nohup
else
  printf 'error: neither screen nor nohup command found\n' >&2
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
LOGS_DIR="$STATE_DIR/logs"
PID_FILE="$STATE_DIR/dispatcher.pid"

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

# ── Lock management ──────────────────────────────────────────────

_lock() {
  local _mode="$1"
  exec 7>"$STATE_DIR/taskq.lock"
  case "$_mode" in
    shared)    flock -s 7 ;;
    exclusive) flock    7 ;;
  esac
}

_unlock() {
  exec 7>&-
}

# ── State directory ──────────────────────────────────────────────

# Create necessary directories and files for the dispatcher state
ensure_state() {
  mkdir -p "$QUEUE_DIR" "$RUN_DIR" "$FAILED_DIR" "$LOGS_DIR"
  [[ -f "$SEQ_FILE" ]] || printf '0\n' >"$SEQ_FILE"
}

# Check if the dispatcher process is alive — no locking, caller must serialize
_dispatcher_running_unsafe() {
  local pid
  [[ -f "$PID_FILE" ]] || return 1
  read -r pid <"$PID_FILE" || return 1
  kill -0 "$pid" 2>/dev/null
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
_cleanup_stale_jobs_unsafe() {
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

  rm -f -- "$RUN_DIR"/.*.pid

  rm -f -- "$QUEUE_DIR"/.*.tmp "$RUN_DIR"/.*.tmp
}

# Enqueue a new job by creating a job file with a unique ID and the command to execute
enqueue_job() {
  _lock exclusive
  _enqueue_job_unsafe "$@"
  _unlock
}

_enqueue_job_unsafe() {
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
      local command_path
      command_path="$(command -v "$target_cmd" 2>/dev/null)" || {
        printf 'error: command not found: %s\n' "$target_cmd" >&2
        exit 127
      }
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

  printf '%s\n' "$job_id"
}

# Check if there are any jobs currently in the queue
queue_has_jobs() {
  local _rc
  _lock shared
  if _queue_has_jobs_unsafe; then
    _rc=0
  else
    _rc=$?
  fi
  _unlock
  return "$_rc"
}

_queue_has_jobs_unsafe() {
  compgen -G "$QUEUE_DIR/*.job" >/dev/null
}

# Run a single job by moving it to the running directory, executing it, and handling success or failure
run_one_job() {
  local job="$1"
  local base job_id claimed failed_path rc job_pid pid_file log_file current_log

  base="$(basename "$job")"
  job_id="${base%.job}"
  claimed="$RUN_DIR/$base"
  failed_path="$FAILED_DIR/${job_id}.failed"
  pid_file="$RUN_DIR/.${job_id}.pid"
  log_file="$LOGS_DIR/${job_id}.log"
  current_log="$LOGS_DIR/current.log"

  # Initialization stage
  _lock exclusive
  if ! mv "$job" "$claimed" 2>/dev/null; then
    _unlock
    return 0
  fi

  {
    printf '# taskq job: %s\n' "$job_id"
    printf '# cmdline: %s\n' "$(extract_cmdline "$claimed")"
    printf '# started: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '#\n'
  } >"$log_file"
  ln -sf "$log_file" "$current_log"

  bash "$claimed" >>"$log_file" 2>&1 &
  job_pid=$!
  printf '%d\n' "$job_pid" >"$pid_file"
  _unlock

  # Wait for job to finish
  set +e
  wait "$job_pid"
  rc=$?
  set -e

  # Clean-up stage
  _lock exclusive

  rm -f "$pid_file"
  {
    head -n 3 "$log_file"
    printf '# finished: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '# exit_code: %d\n' "$rc"
    tail -n +4 "$log_file"
  } >"${log_file}.tmp" && mv "${log_file}.tmp" "$log_file"

  rm -f "$current_log"

  if (( rc == 0 )); then
    rm -f "$claimed"
  else
    mv "$claimed" "$failed_path"
    printf 'job failed: %s (exit=%d)\n' "$job_id" "$rc" >&2
  fi
  _unlock

  return 0
}

# Check under LOCK whether to exit after idle timeout.
# Returns 0 (true) when jobs have appeared — caller should reset idle timer and continue.
# Exits the process when the queue is confirmed empty.
_try_exit_dispatcher() {
  _lock exclusive
  if _queue_has_jobs_unsafe; then
    _unlock
    return 0
  fi
  rm -f "$PID_FILE"
  _unlock
  exit 0
}

# Main loop for the dispatcher that continuously checks for new jobs and executes them, with an idle timeout for automatic shutdown
dispatcher_loop() {
  local idle_timeout="$1"
  local idle_since=0
  local found
  local job
  local now

  ensure_state

  printf '%s\n' "$$" >"$PID_FILE"

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
      if _try_exit_dispatcher; then
        idle_since=0
        continue
      fi
    fi

    sleep "$POLL_INTERVAL"
  done
}

_start_dispatcher_unsafe() {
  local idle_timeout="$1"

  if _dispatcher_running_unsafe; then
    return 0
  fi

  _cleanup_stale_jobs_unsafe

  if [[ "$BACKEND" == screen ]]; then
    (
      exec 7>&-
      exec </dev/null >/dev/null 2>&1
      exec screen -dmS "$SESSION_NAME" \
        bash "$SELF_PATH" --dispatcher --idle-timeout "$idle_timeout"
    )
  else
    (
      exec 7>&-
      exec </dev/null >/dev/null 2>&1
      nohup bash "$SELF_PATH" --dispatcher --idle-timeout "$idle_timeout" &
    )
  fi

  local waited=0
  while ! _dispatcher_running_unsafe && (( waited < 5 )); do
    sleep 1
    waited=$((waited + 1))
  done

  if ! _dispatcher_running_unsafe; then
    printf 'error: failed to start dispatcher with %s\n' "$BACKEND" >&2
    exit 1
  fi
}

submit_job() {
  local idle_timeout="$1"
  shift

  ensure_state
  _lock exclusive

  _start_dispatcher_unsafe "$idle_timeout"
  SUBMITTED_JOB_ID="$(_enqueue_job_unsafe "$@")"

  _unlock
}

show_status() {
  ensure_state

  _lock shared
  local is_running=false
  local dispatcher_pid='-'
  if _dispatcher_running_unsafe; then
    is_running=true
    read -r dispatcher_pid <"$PID_FILE" 2>/dev/null || dispatcher_pid='-'
  fi
  local queued_files=("$QUEUE_DIR"/*.job)
  local running_files=("$RUN_DIR"/*.job)
  local failed_files=("$FAILED_DIR"/*.failed)
  local current_cmdline=''
  local upcoming_cmdline=''
  if (( ${#running_files[@]} > 0 )); then
    current_cmdline="$(extract_cmdline "${running_files[0]}")"
  fi
  if (( ${#queued_files[@]} > 0 )); then
    upcoming_cmdline="$(extract_cmdline "${queued_files[0]}")"
  fi
  _unlock

  local queued="${#queued_files[@]}"
  local failed="${#failed_files[@]}"

  local bold_green=$'\033[1;32m'
  local bold_red=$'\033[1;31m'
  local bold_yellow=$'\033[1;33m'
  local gray=$'\033[90m'
  local reset=$'\033[0m'

  if $is_running; then
    printf 'Dispatcher: %sRUNNING%s (backend: %s, pid: %s)\n' \
      "$bold_green" "$reset" "$BACKEND" "$dispatcher_pid"
  else
    printf 'Dispatcher: %sSTOPPED%s\n' "$bold_red" "$reset"
  fi

  printf '%sScript path: %s%s\n' "$gray" "$SELF_PATH" "$reset"
  printf '%sState dir: %s%s\n' "$gray" "$STATE_DIR" "$reset"

  if $is_running; then
    if [[ -n "$current_cmdline" ]]; then
      printf '%sCurrent job%s: %s\n' "$bold_green" "$reset" "$current_cmdline"

      local current_log="$LOGS_DIR/current.log"
      if [[ -f "$current_log" ]]; then
        printf '%s--- last 20 lines (%s) ---%s\n' "$gray" "$current_log" "$reset"
        tail -n +5 "$current_log" | tail -20
        printf '%s────────────────────────────────────────%s\n' "$gray" "$reset"
      fi
    else
      printf '%sNo running jobs%s\n' "$bold_yellow" "$reset"
    fi

    if [[ -n "$upcoming_cmdline" ]]; then
      printf '%sUpcoming job%s: %s\n' "$bold_yellow" "$reset" "$upcoming_cmdline"
    fi

    printf '%sQueued jobs: %s%s\n' "$bold_yellow" "$queued" "$reset"
    printf '%sFailed jobs: %s%s\n' "$bold_red" "$failed" "$reset"
  fi
}

handle_kill() {
  local force="${1:-false}"

  ensure_state

  local bold_red=$'\033[1;31m'
  local reset=$'\033[0m'

  local lock_mode=shared
  $force && lock_mode=exclusive

  _lock "$lock_mode"
  if ! _dispatcher_running_unsafe; then
    _unlock
    printf 'Dispatcher is not running; nothing to do.\n'
    return 0
  fi

  printf 'Trying to stop dispatcher (backend: %s)\n' "$BACKEND"

  local -a queued_files=("$QUEUE_DIR"/*.job)
  local -a running_files=("$RUN_DIR"/*.job)
  local queued="${#queued_files[@]}"
  local running="${#running_files[@]}"

  local -a running_cmdlines=()
  local f
  for f in "${running_files[@]}"; do
    running_cmdlines+=("$(extract_cmdline "$f")")
  done

  if $force; then
    printf 'Running task(s) will be interrupted:\n'
    for f in "${running_cmdlines[@]}"; do
      printf '  - %s\n' "$f"
    done
    printf 'Queued jobs: %d (will be discarded)\n' "$queued"
    _stop_dispatcher_unsafe
    _unlock
    return 0
  fi

  if (( running == 0 )); then
    _unlock
    _lock exclusive
    _stop_dispatcher_unsafe
    _unlock
    return 0
  fi
  _unlock

  printf '%sWARNING: dispatcher is currently running task(s). Killing it will interrupt them immediately.%s\n' "$bold_red" "$reset"
  printf '%sInterrupted or queued tasks will not be resumed automatically after termination.%s\n' "$bold_red" "$reset"
  printf 'Running task(s):\n'
  for f in "${running_cmdlines[@]}"; do
    printf '  - %s\n' "$f"
  done
  printf 'Queued jobs: %d\n' "$queued"

  printf 'Terminate dispatcher? [y/N] '
  local reply
  if ! read -r reply; then
    printf '\nKill cancelled\n'
    return 1
  fi

  case "$reply" in
    [Yy]|[Yy][Ee][Ss])
      _lock exclusive
      _stop_dispatcher_unsafe
      _unlock
      ;;
    *)
      printf 'Kill cancelled\n'
      ;;
  esac
}

stop_dispatcher() {
  _lock exclusive
  _stop_dispatcher_unsafe
  _unlock
}

_stop_dispatcher_unsafe() {
  if _dispatcher_running_unsafe; then
    local pid jpid pidf
    read -r pid <"$PID_FILE"

    for pidf in "$RUN_DIR"/.*.pid; do
      [[ -f "$pidf" ]] || continue
      read -r jpid <"$pidf" 2>/dev/null && kill "$jpid" 2>/dev/null || true
      rm -f "$pidf"
    done

    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

usage() {
  cat <<EOF
usage:
  $0 <command> [args...]
  $0 --status
  $0 --kill
  $0 --kill-yes
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
      handle_kill false
      ;;
    --kill-yes)
      handle_kill true
      ;;
    --help|-h)
      usage
      ;;
    "")
      show_status
      echo
      usage
      exit 0
      ;;
    *)
      submit_job "$DEFAULT_IDLE_TIMEOUT" "$@"
      printf 'queued job: %s\n' "$SUBMITTED_JOB_ID"
      printf 'dispatcher backend: %s\n' "$BACKEND"
      ;;
  esac
}

main "$@"
