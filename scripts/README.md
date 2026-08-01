# scripts

Miscellaneous utility scripts.

| Script | Language | Description |
|--------|----------|-------------|
| `.bash_aliases` | Bash | Linux process inspection tools: `pinfo`, `pstats`, `pginfo`, `pcinfo`, `showmem` |
| `catlsa.py` | Python | Transparent `cat`/`ls` wrapper with archive support (read-only) |
| `gather.sh` | Bash | Parallel command executor with concurrency control |
| `run_nb.py` | Python | Execute Jupyter notebooks in parallel with frequency control |
| `taskq.sh` | Bash | Task queue dispatcher with `screen`/`nohup` backend |
| `vima.sh` | Bash | Transparent `vim` wrapper with archive writeback |

---

### .bash_aliases

Linux-only shell aliases and functions for process inspection.

```bash
# Script aliases (all platforms)
catlsa               # → catlsa.py
gather               # → gather.sh
run_nb               # → run_nb.py
taskq                # → taskq.sh
vima                 # → vima.sh

# Process inspection (Linux only)
join <pid> [pid...]  # wait for processes to exit (PID-reuse safe)
showmem              # RSS summary by user
memtop [user]        # top-20 processes by RSS for a user (default: huangfusl)

pinfo <pid> [pid...] # detailed per-process info, color-coded CPU/MEM/RSS/VSZ
pstats <pid> [sec]   # sample a PID for N seconds, stats summary (mean/max/min/std)
pginfo <pid>         # process tree view with DFS walk, target PID highlighted
pcinfo <pid>         # direct children of a PID
```

### catlsa.py

Transparent `cat`/`ls` wrapper with archive support. Walks a path left-to-right, resolving archives (tar, zip, 7z, gz, bz2, xz) and auto-dispatching the leaf: file → `cat`, directory → `ls`. Only necessary files are extracted — read-only, archives are never modified.

```bash
catlsa project.tar.gz/src/main.py          # cat the file
catlsa project.tar.gz/src/                 # list the directory
catlsa outer.zip/inner.tar.gz/config.yml   # nested archives
catlsa data.tar.gz                         # list root of archive
```

### gather.sh

Run multiple shell commands in parallel with optional concurrency limits.

```bash
gather.sh "echo hello" "echo world"
gather.sh --max-concurrent 4 --fail-fast "cmd1" "cmd2" "cmd3"
gather.sh --show-log 3 run.log           # inspect output of instance 3
```

#### Reading Logs

Each command's output is tagged with an instance number. Filter with `grep`:

```bash
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
```

### run_nb.py

Execute multiple instances of a Jupyter notebook in parallel with configurable concurrency and start-interval control. Requires `nbformat` and `nbclient`.

```bash
# Run 10 instances of a notebook, 4 at a time, starting one every 30s
run_nb.py -f notebook.ipynb -t 10 -w 4 -i 30

# Single run with a 5-minute timeout and per-instance output recording
run_nb.py -f notebook.ipynb --timeout 300 --record-output

# Ignore cell errors (continue execution instead of raising)
run_nb.py -f notebook.ipynb -t 5 -w 2 --ignore-errors
```

Each instance gets a copy of the template (e.g. `notebook_1.ipynb`, `notebook_2.ipynb`, …) and runs in its own kernel with `AUTOMATED=1` set in the environment.

**Note**: The script creates a temporary `.{notebook_name}_run_nb.lock` file to prevent concurrent runs on the same notebook. If a previous run exits abnormally, delete the lock file before running again.

### taskq.sh

Task queue dispatcher. Enqueue commands and let a background dispatcher run them sequentially. Uses `screen` or `nohup` as the backend.

```bash
scripts/taskq.sh python job.py           # enqueue a job
scripts/taskq.sh --status                # show queue state
scripts/taskq.sh --show-log 42           # view job output
scripts/taskq.sh --kill                  # stop the dispatcher
```

### vima.sh

Transparent `vim` wrapper with archive support. Walks a path left-to-right, resolving archives (tar, zip, 7z, gz, bz2, xz), opens the leaf file in `vim`, and **rebuilds** modified archives on exit. Supports passing vim flags via `--`.

```bash
vima project.tar.gz/src/main.py           # edit, archive rebuilt on exit
vima -R -- outer.zip/inner.tar.gz/config.yml  # read-only flag
```

Caveats: archives are always rebuilt (even if no changes made). No metadata preservation, no transactional guarantee.
