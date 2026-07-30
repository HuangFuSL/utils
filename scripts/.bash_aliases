# ── Script aliases (all platforms) ─────────────────────────────────────
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _BASH_ALIASES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  alias gather="$_BASH_ALIASES_DIR/gather.sh"
  alias run_nb="python3 $_BASH_ALIASES_DIR/run_nb.py"
  alias taskq="$_BASH_ALIASES_DIR/taskq.sh"
  alias vima="$_BASH_ALIASES_DIR/vima.sh"
  alias catlsa="python3 $_BASH_ALIASES_DIR/catlsa.py"
fi

# ── join: wait for PIDs to exit ─────────────────────────────────────────
join() {
  local pids=("$@")
  ((${#pids[@]})) || { echo "Usage: join <pid> [pid...]" >&2; return 1; }
  local -A starttimes
  local total=${#pids[@]}
  local remain=("${pids[@]}")
  local done=()

  # snapshot starttime for each PID
  local pid st
  for pid in "${pids[@]}"; do
    st=$(awk '{print $22}' /proc/"$pid"/stat 2>/dev/null) || {
      echo "[join] PID $pid does not exist" >&2
      continue
    }
    starttimes[$pid]=$st
  done

  while ((${#remain[@]})); do
    local new_remain=()
    for pid in "${remain[@]}"; do
      local current_st
      current_st=$(awk '{print $22}' /proc/"$pid"/stat 2>/dev/null) || { done+=("$pid"); continue; }
      if [[ "${starttimes[$pid]}" == "$current_st" ]]; then
        new_remain+=("$pid")
      else
        done+=("$pid")  # PID recycled — original process is dead
      fi
    done
    remain=("${new_remain[@]}")
    if ((${#done[@]})); then
      printf '[join] %s exited\n' "${done[*]}"
      done=()
    fi
    ((${#remain[@]})) && sleep 1
  done
  echo "[join] all $total process(es) exited"
}

# Only load on Linux — /proc, GNU ps flags, etc.
if [[ "$(uname)" != "Linux" ]]; then
  echo ".bash_aliases: skipping (Linux only, running on $(uname))" >&2
  return 0 2>/dev/null || exit 0
fi

alias stats=$'awk \'{x=$1; s+=x; s2+=x*x; n++; if(n==1){min=x;max=x}; if(x<min) min=x; if(x>max) max=x} END {mean=s/n; var=s2/n-mean^2; std=sqrt(var); printf "n    = %d\\nsum  = %.4f\\nmean = %.4f\\nmin  = %.4f  \\nmax  = %.4f\\nvar  = %.4f\\nstd  = %.4f\\n",n,s,mean,min,max,var,std}\' "$@"'
alias showmem='ps -eo user:20,rss --no-headers | awk '\''{mem[$1]+=$2} END {for (u in mem) printf "%-20s %10.2f MB\n", u, mem[u]/1024}'\'' | sort -k2 -rn'
memtop() {
  local user="${1:-huangfusl}"
  ps -u "$user" -o pid,rss,comm --no-headers | sort -k2 -rn | awk '{printf "%-8s %10.2f MB  %s\n", $1, $2/1024, $3}' | head -20
}

# shared awk color functions
AWK_COLORS='function cpu_ansi(pct) {
  if (pct < 10)  return "\033[90m"
  if (pct < 50)  return "\033[32m"
  if (pct < 100) return "\033[33m"
  if (pct < 200) return "\033[38;5;214m"
  return "\033[31m"
}
function rss_ansi(kb) {
  if (kb < 102400)  return "\033[90m"
  if (kb < 512000)  return "\033[32m"
  if (kb < 2097152) return "\033[33m"
  if (kb < 8388608) return "\033[38;5;214m"
  return "\033[31m"
}
function cpu_fmt(v)           { return cpu_ansi(v)  sprintf("%5.1f%%", v)      "\033[0m" }
function mem_fmt(v, r)        { return rss_ansi(r)  sprintf("%5.1f%%", v)      "\033[0m" }
function rss_fmt(v, d, u)     { return rss_ansi(v)  sprintf("%7.1f%s", v/d, u) "\033[0m" }
function vsz_fmt(v, d, u)     { return sprintf("%7.1f%s", v/d, u) }
'

pinfo() {
  local tw="${COLUMNS:-$(tput cols 2>/dev/null)}"
  tw="${tw:-120}"

  # -- pids from args, or stdin if no args --
  local pids=() pid
  if [[ $# -gt 0 ]]; then
    pids=("$@")
  else
    while IFS=$' \t\n' read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done
  fi
  ((${#pids[@]})) || { echo "Usage: pinfo <pid> [pid...]"; return 1; }

  # -- header --
  printf "%-8s %-8s %-12s %6s %6s %9s %9s %10s %s\n" PID PPID USER CPU% MEM% RSS VSZ TIME COMMAND

  for pid in "${pids[@]}"; do
    local line
    line=$(ps -p "$pid" -o pid=,ppid=,user:12=,%cpu=,%mem=,rss=,vsz=,time=,args= 2>/dev/null) || { echo "[$pid] not found"; continue; }
    echo "$line" | awk -v tw="$tw" "$AWK_COLORS"$'\n''
    function vlen(s) { gsub(/\033\[[0-9;]*m/, "", s); return length(s) }

    function fmt_size(kb) {
      mb = kb / 1024
      if (mb >= 10240) return sprintf("%7.1fGB", mb / 1024)
      else             return sprintf("%7.1fMB", mb)
    }

    {
      sub("-", "d", $8)  # 4-13:23:41 → 4d13:23:41
      cpu_str = cpu_fmt($4)
      rc = rss_ansi($6)
      mem_str = rc sprintf("%5.1f%%", $5) "\033[0m"
      rss_str = rc fmt_size($6) "\033[0m"
      vsz_str = fmt_size($7)
      prefix = sprintf("%-8s %-8s %-12s %s %s %s %s %10s ",
                       $1, $2, $3, cpu_str, mem_str, rss_str, vsz_str, $8)
      pfx_len = vlen(prefix)

      cmd = ""
      for (i = 9; i <= NF; i++) cmd = cmd $i (i < NF ? " " : "")

      cw = tw - pfx_len
      if (cw < 20) cw = tw - 10
      pos = 1
      first = 1
      while (pos <= length(cmd)) {
        seg = substr(cmd, pos, cw)
        if (first) { printf "%s%s\n", prefix, seg; first = 0 }
        else        printf "%*s%s\n", pfx_len, "", seg
        pos += cw
      }
    }'
  done
}

pstats() {
  local pid="${1:?Usage: pstats <pid> [seconds]}"
  local sec="${2:-10}"

  [[ -d /proc/$pid ]] || { echo "PID $pid not found"; return 1; }

  # command line
  local cmd
  cmd=$(tr '\0' ' ' 2>/dev/null < /proc/$pid/cmdline)
  [[ -z "$cmd" ]] && cmd=$(ps -p "$pid" -o comm= 2>/dev/null)

  # CPU time: utime + stime from /proc/pid/stat
  local clk_tck utime stime
  clk_tck=$(getconf CLK_TCK 2>/dev/null || echo 100)
  read -r _ _ _ _ _ _ _ _ _ _ _ _ _ utime stime _ <<< \
    $(sed 's/^[0-9]* (.*) //' /proc/$pid/stat)
  local start_ticks=$((utime + stime))
  local prev_ticks=$start_ticks
  local end_ticks=$start_ticks

  # sample loop — CPU% from /proc/pid/stat delta, MEM from ps
  local tmpfile i cpu_pct
  tmpfile=$(mktemp)

  for ((i = 0; i < sec; i++)); do
    sleep 1

    read -r _ _ _ _ _ _ _ _ _ _ _ _ _ utime stime _ <<< \
    $(sed 's/^[0-9]* (.*) //' /proc/$pid/stat) 2>/dev/null \
      || { echo "Process $pid terminated after $i samples"; break; }
    end_ticks=$((utime + stime))
    cpu_pct=$(awk -v d=$((end_ticks - prev_ticks)) -v c="$clk_tck" \
      'BEGIN { printf "%.1f", d / c * 100 }')
    prev_ticks=$end_ticks

    local mem_line
    mem_line=$(ps -p "$pid" -o %mem=,rss=,vsz= --no-headers 2>/dev/null)
    [[ -z "$mem_line" ]] && { echo "Process $pid terminated after $i samples"; break; }

    echo "$cpu_pct $mem_line" >> "$tmpfile"
  done

  # header: CPU time = total ticks delta / CLK_TCK
  local cpu_time
  cpu_time=$(awk -v s="$start_ticks" -v e="$end_ticks" -v c="$clk_tck" \
    'BEGIN { printf "%.1f", (e - s) / c }')

  # header
  local n_samples
  n_samples=$(wc -l < "$tmpfile")
  echo "PID $pid | $cmd"
  echo "Duration: ${n_samples}s | CPU time: ${cpu_time}s"
  echo

  # -- stats table --
  if [[ -s "$tmpfile" ]]; then
    awk "$AWK_COLORS"$'\n''
     {
      cpu[NR] = $1 + 0; mem[NR] = $2 + 0; rss[NR] = $3 + 0; vsz[NR] = $4 + 0
      c_sum += $1; m_sum += $2; r_sum += $3; v_sum += $4
      if (NR == 1) {
        c_max = c_min = $1; m_max = m_min = $2
        r_max = r_min = $3; v_max = v_min = $4
      } else {
        if ($1 > c_max) c_max = $1; if ($1 < c_min) c_min = $1
        if ($2 > m_max) m_max = $2; if ($2 < m_min) m_min = $2
        if ($3 > r_max) r_max = $3; if ($3 < r_min) r_min = $3
        if ($4 > v_max) v_max = $4; if ($4 < v_min) v_min = $4
      }
    }
    END {
      n = NR; if (n == 0) exit
      c_mean = c_sum/n; m_mean = m_sum/n; r_mean = r_sum/n; v_mean = v_sum/n

      # stddev
      for (i=1; i<=n; i++) {
        csq += (cpu[i]-c_mean)^2; msq += (mem[i]-m_mean)^2
        rsq += (rss[i]-r_mean)^2; vsq += (vsz[i]-v_mean)^2
      }
      c_std = sqrt(csq/n); m_std = sqrt(msq/n); r_std = sqrt(rsq/n); v_std = sqrt(vsq/n)

      # auto-unit for RSS / VSZ
      if (r_max / 1024 >= 10240) { r_div = 1048576; r_unit = "GB" }
      else                        { r_div =    1024; r_unit = "MB" }
      if (v_max / 1024 >= 10240) { v_div = 1048576; v_unit = "GB" }
      else                        { v_div =    1024; v_unit = "MB" }

      printf "%6s %6s %6s %9s %9s\n", "", "CPU%", "MEM%", "RSS", "VSZ"
      printf "%6s %s %s %s %s\n", "mean",
        cpu_fmt(c_mean),       mem_fmt(m_mean, r_mean), rss_fmt(r_mean, r_div, r_unit), vsz_fmt(v_mean, v_div, v_unit)
      printf "%6s %s %s %s %s\n", "max",
        cpu_fmt(c_max),        mem_fmt(m_max,  r_max),  rss_fmt(r_max,  r_div, r_unit), vsz_fmt(v_max,  v_div, v_unit)
      printf "%6s %s %s %s %s\n", "min",
        cpu_fmt(c_min),        mem_fmt(m_min,  r_min),  rss_fmt(r_min,  r_div, r_unit), vsz_fmt(v_min,  v_div, v_unit)
      printf "%6s %5.1f%% %5.1f%% %7.1f%s %7.1f%s\n",
        "std", c_std, m_std, r_std/r_div, r_unit, v_std/v_div, v_unit
    }
    ' "$tmpfile"
  fi
  rm -f "$tmpfile"
}

# ---- pginfo / pcinfo ----

# DFS walk: output "depth|bars|pid|ppid|has_children|is_target"
_walk() {
  local pid=$1 depth=$2 target=$3 bars=$4
  [[ -d /proc/$pid ]] || return

  local children_list
  children_list=$(tr ' ' '\n' < "/proc/$pid/task/$pid/children" 2>/dev/null | sort -n)
  local children=() child
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    # skip self unless self is the target
    [[ "$child" == "$$" && "$child" != "$target" ]] && continue
    children+=("$child")
  done <<< "$children_list"
  local n_children=${#children[@]}
  local has_children=0; ((n_children > 0)) && has_children=1

  # parent PPID from /proc/pid/stat (field 4)
  local ppid
  ppid=$(sed 's/^[0-9]* (.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $2}')
  ppid="${ppid:-0}"

  local is_target=0; [[ "$pid" == "$target" ]] && is_target=1
  echo "$depth|$bars|$pid|$ppid|$has_children|$is_target"

  local i=0 remaining child_bars
  for child in "${children[@]}"; do
    ((i++))
    remaining=$((n_children - i))
    if ((remaining > 0)); then child_bars="${bars}1"; else child_bars="${bars}0"; fi
    _walk "$child" $((depth + 1)) "$target" "$child_bars"
  done
}

pginfo() {
  local target="${1:?Usage: pginfo <pid>}"
  [[ -d /proc/$target ]] || { echo "PID $target not found"; return 1; }

  # walk up to root
  local pid=$target ppid
  while ppid=$(sed 's/^[0-9]* (.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $2}') && \
        [[ -n "$ppid" ]] && ((ppid > 1)); do
    pid=$ppid
  done
  local root=$pid

  # DFS walk → tree_data
  local tree_data
  tree_data=$(mktemp)
  _walk "$root" 0 "$target" "" > "$tree_data"

  # compute max depth & tree width
  local max_depth
  max_depth=$(awk -F'|' '{if ($1+0 > m) m=$1+0} END {print m}' "$tree_data")
  local tw=$(( max_depth * 2 + 1 ))

  # collect PIDs and fetch pinfo data
  local pids=() p
  while IFS='|' read -r _ _ p _ _ _; do pids+=("$p"); done < "$tree_data"

  # header
  printf "%-${tw}s %-8s %-8s %-12s %6s %6s %9s %9s %10s %s\n" \
    TREE PID PPID USER CPU% MEM% RSS VSZ TIME COMMAND

  # terminal width for command wrapping
  local tw_term="${COLUMNS:-$(tput cols 2>/dev/null)}"
  tw_term="${tw_term:-120}"

  # process each node
  local idx=0
  while IFS='|' read -r depth bars pid ppid has_children is_target; do
    local line
    line=$(ps -p "$pid" -o pid=,ppid=,user:12=,%cpu=,%mem=,rss=,vsz=,time=,args= --no-headers 2>/dev/null)
    if [[ -z "$line" ]]; then
      echo "$depth ${bars:--} $has_children $is_target $pid $ppid _DEAD_ 0 0 0 0 00:00:00 <exited>"
    else
      echo "$depth ${bars:--} $has_children $is_target $line"
    fi
  done < "$tree_data" | awk -v tw="$tw" -v tw_term="$tw_term" "$AWK_COLORS"$'\n''

  function tree_col(depth, bars, has_children, tw,   s, l, conn, dw) {
    s = ""
    # indent: bars from ancestor levels
    for (l = 1; l < depth; l++) {
      if (substr(bars, l, 1) == "1") s = s "│ "
      else                           s = s "  "
    }
    if (depth == 0) {
      s = s "▼"
      dw = 1
    } else {
      conn = (substr(bars, depth, 1) == "1") ? "├─" : "└─"
      s = s conn (has_children ? "▼" : "─")
      dw = depth * 2 + 1
    }
    while (dw < tw) { s = s "─"; dw++ }
    return s
  }

  function cont_tree_col(tc) {
    gsub(/├/, "│", tc)
    gsub(/└/, " ", tc)
    gsub(/▼/, "│", tc)
    gsub(/─/, " ", tc)
    return tc
  }

  {
    depth    = $1 + 0
    bars     = ($2 == "-") ? "" : $2
    has_child= $3 + 0
    is_target= $4 + 0
    pid      = $5; ppid = $6; user = $7
    dead     = (user == "_DEAD_") ? 1 : 0
    cpu_pct  = $8 + 0; mem_pct = $9 + 0
    rss_kb   = $10 + 0; vsz_kb = $11 + 0
    cputime  = $12
    if (!dead) sub("-", "d", cputime)
    cmd = ""
    for (i = 13; i <= NF; i++) cmd = cmd $i (i < NF ? " " : "")

    tc = tree_col(depth, bars, has_child, tw)

    if (dead) {
      cpu_str = sprintf("%6s", "-")
      mem_str = sprintf("%6s", "-")
      rss_str = sprintf("%9s", "-")
      vsz_str = sprintf("%9s", "-")
      user = "dead"
    } else {
      cpu_str = cpu_fmt(cpu_pct)
      rc = rss_ansi(rss_kb)
      mem_str = rc sprintf("%5.1f%%", mem_pct) "\033[0m"
      r_div = 1024; r_unit = "MB"
      v_div = 1024; v_unit = "MB"
      if (rss_kb / 1024 >= 10240) { r_div = 1048576; r_unit = "GB" }
      if (vsz_kb / 1024 >= 10240) { v_div = 1048576; v_unit = "GB" }
      rss_str = rc sprintf("%7.1f%s", rss_kb/r_div, r_unit) "\033[0m"
      vsz_str = sprintf("%7.1f%s", vsz_kb/v_div, v_unit)
    }

    bold_on = ""; bold_off = ""
    if (dead) {
      bold_on = "\033[90;3m"; bold_off = "\033[0m"
      user = "dead"
    } else if (is_target) {
      bold_on = "\033[7m"; bold_off = "\033[0m"
      gsub(/\033\[0m/, "\033[0;7m", cpu_str)
      gsub(/\033\[0m/, "\033[0;7m", mem_str)
      gsub(/\033\[0m/, "\033[0;7m", rss_str)
    }

    # prefix = everything before command, excluding bold
    prefix = sprintf("%s %-8s %-8s %-12s %s %s %s %s %10s ",
              tc, pid, ppid, user, cpu_str, mem_str, rss_str, vsz_str, cputime)
    # cw: command width in display chars (tw + 78 = tree + data cols + separators)
    cw = tw_term - (tw + 78)
    if (cw < 20) cw = tw_term - 10

    data_pad = 78

    pos = 1
    first = 1
    while (pos <= length(cmd)) {
      seg = substr(cmd, pos, cw)
      if (first) {
        printf "%s%s%s%s\n", bold_on, prefix, seg, bold_off
        first = 0
      } else {
        ctc = cont_tree_col(tc)
        printf "%s%s%*s%s%s\n", bold_on, ctc, data_pad, "", seg, bold_off
      }
      pos += cw
    }
  }'

  rm -f "$tree_data"
}

pcinfo() {
  local pid="${1:?Usage: pcinfo <pid>}"
  local children
  children=$(pgrep -aP "$pid" 2>/dev/null)
  [[ -z "$children" ]] && return 0

  printf "%-8s %-8s %-12s %6s %6s %9s %9s %10s %s\n" \
    PID PPID USER CPU% MEM% RSS VSZ TIME COMMAND

  pgrep -P "$pid" 2>/dev/null | while read -r cpid; do
    local line
    line=$(ps -p "$cpid" -o pid=,ppid=,user:12=,%cpu=,%mem=,rss=,vsz=,time=,args= --no-headers 2>/dev/null)
    [[ -z "$line" ]] && continue
    echo "$line" | awk "$AWK_COLORS"$'\n''
    {
      sub("-", "d", $8)
      cpu_str = cpu_ansi($4 + 0) sprintf("%5.1f%%", $4 + 0) "\033[0m"
      rc = rss_ansi($6 + 0)
      mem_str = rc sprintf("%5.1f%%", $5 + 0) "\033[0m"
      r_div = 1024; r_unit = "MB"
      v_div = 1024; v_unit = "MB"
      if ($6 / 1024 >= 10240) { r_div = 1048576; r_unit = "GB" }
      if ($7 / 1024 >= 10240) { v_div = 1048576; v_unit = "GB" }
      rss_str = rc sprintf("%7.1f%s", $6/r_div, r_unit) "\033[0m"
      vsz_str = sprintf("%7.1f%s", $7/v_div, v_unit)
      cmd = ""
      for (i = 9; i <= NF; i++) cmd = cmd $i (i < NF ? " " : "")
      printf "%-8s %-8s %-12s %s %s %s %s %10s %s\n",
        $1, $2, $3, cpu_str, mem_str, rss_str, vsz_str, $8, cmd
    }'
  done
}