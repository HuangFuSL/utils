#!/usr/bin/env bash
# vima.sh — transparent vim wrapper with archive support. see --help for details.
set -euo pipefail

readonly MAX_NESTING=16
TEMPDIRS=()
TMPFILES=()
_VIM_RC=0

cleanup() {
    local d
    for d in "${TEMPDIRS[@]:-}"; do [[ -n "$d" ]] && rm -rf "$d" 2>/dev/null || true; done
    for f in "${TMPFILES[@]:-}"; do [[ -n "$f" ]] && rm -f  "$f" 2>/dev/null || true; done
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
usage: vima [vim-flags] [--] <path> [more-vim-args]

Transparent vim wrapper with archive support. Walks the path left-to-right,
resolving archives (tar, zip, 7z, gz, bz2, xz) and opening the leaf file in vim.
Modified archives are rebuilt in full. Bare paths fall through to vim directly.

Use -- to separate vim flags from the path explicitly:
  vima -R -- project.tar.gz/src/main.py
Without --, the first non-flag argument is taken as the path.
With no arguments, this help is shown.

examples:
  vima file.txt
  vima project.tar.gz/src/main.py
  vima -R outer.zip/inner.tar.gz/config.yml
  vima data.zip.gz/inner/file.txt

caveats:
  - the script assumes modification — archives are always rebuilt, even if
    no changes were made. for read-only viewing, see cata.sh.
  - the entire archive is rebuilt — all changes in the extracted tree,
    including files edited via :e, are preserved.
  - no path-traversal safety. validate archives before use.
  - no metadata preservation (permissions, timestamps, symlinks).
  - no transactional guarantee.
  - no recovery on writeback failure.
EOF
}

ensure_tool() {
    command -v "$1" >/dev/null 2>&1 || { echo "error: $1 not found" >&2; exit 1; }
}


detect_format() {
    local path="$1"
    local name

    [[ -f "$path" ]] || return 1

    local mime
    mime=$(file -b --mime-type "$path" 2>/dev/null) || return 1
    case "$mime" in
        application/zip)
            echo "zip"
            return 0
            ;;
        application/x-tar)
            echo "tar"
            return 0
            ;;
        application/gzip)
            case "$(file -zb "$path" 2>/dev/null)" in
                *tar*) echo "tar.gz"; return 0 ;;
                *)     echo "gz";    return 0 ;;
            esac
            ;;
        application/x-bzip2)
            case "$(file -zb "$path" 2>/dev/null)" in
                *tar*) echo "tar.bz2"; return 0 ;;
                *)     echo "bz2";    return 0 ;;
            esac
            ;;
        application/x-xz)
            case "$(file -zb "$path" 2>/dev/null)" in
                *tar*) echo "tar.xz"; return 0 ;;
                *)     echo "xz";    return 0 ;;
            esac
            ;;
        application/x-7z-compressed)
            echo "7z"
            return 0
            ;;
    esac

    return 1
}

validate_archive() {
    local a="$1"
    [[ -f "$a" ]] || { echo "error: archive not found: $a" >&2; exit 1; }
    [[ -r "$a" ]] || { echo "error: archive not readable: $a" >&2; exit 1; }
}

# helper: exec vim with optional flags, safe on bash 3.2 empty arrays

vim_call() {
    local -a args=()
    if (( ${#VIM_PRE[@]} )); then args+=("${VIM_PRE[@]}"); fi
    args+=("$@")
    if (( ${#VIM_POST[@]} )); then args+=("${VIM_POST[@]}"); fi
    vim "${args[@]}"
}

stem() { local n=$(basename "$1") s="$2"; echo "${n%.$s}"; }

extract_all() {
    local archive="$1" tmpdir="$2" format="$3"
    case "$format" in
        tar|tar.gz|tar.bz2|tar.xz)
            ensure_tool tar
            tar xf "$archive" -C "$tmpdir"
            ;;
        zip)
            ensure_tool unzip;
            ensure_tool zip;
            unzip -q -o "$archive" -d "$tmpdir"
            ;;
        7z)
            ensure_tool 7z;
            7z x -o"$tmpdir" "$archive" >/dev/null
            ;;
        gz)
            ensure_tool gzip;
            gunzip -c "$archive" > "$tmpdir/$(stem "$archive" gz)"
            ;;
        bz2)
            ensure_tool bzip2;
            bunzip2 -c "$archive" > "$tmpdir/$(stem "$archive" bz2)"
            ;;
        xz)
            ensure_tool xz;
            unxz -c "$archive" > "$tmpdir/$(stem "$archive" xz)"
            ;;
    esac
}

write_back() {
    local archive="$1" tmpdir="$2" format="$3"
    local tmp
    tmp=$(mktemp "$(dirname "$archive")/vima.XXXXXX") || { echo "error: cannot create temp file" >&2; exit 1; }
    TMPFILES+=("$tmp")
    local ok=0
    case "$format" in
        tar)      (cd "$tmpdir" && tar cf  "$tmp" .) && ok=1 ;;
        tar.gz)   (cd "$tmpdir" && tar czf "$tmp" .) && ok=1 ;;
        tar.bz2)  (cd "$tmpdir" && tar cjf "$tmp" .) && ok=1 ;;
        tar.xz)   (cd "$tmpdir" && tar cJf "$tmp" .) && ok=1 ;;
        zip)      ensure_tool zip;  rm -f "$tmp"; (cd "$tmpdir" && zip -rq "$tmp" .) && ok=1 ;;
        7z)       ensure_tool 7z;   rm -f "$tmp"; (cd "$tmpdir" && 7z a "$tmp" . >/dev/null) && ok=1 ;;
        gz)       ensure_tool gzip;   gzip   -c "$tmpdir/$(stem "$archive" gz)"   > "$tmp" && ok=1 ;;
        bz2)      ensure_tool bzip2;  bzip2  -c "$tmpdir/$(stem "$archive" bz2)"  > "$tmp" && ok=1 ;;
        xz)       ensure_tool xz;     xz     -c "$tmpdir/$(stem "$archive" xz)"   > "$tmp" && ok=1 ;;
    esac
    if (( ok )); then
        mv "$tmp" "$archive" || { echo "error: cannot replace $archive" >&2; exit 1; }
    else
        echo "error: writeback failed" >&2; exit 1
    fi
}

walk() {
    local root="$1" path="$2" depth="${3:-0}"

    if [[ "$path" == /* ]]; then
        root="/"
        path="${path#/}"
    fi

    (( depth <= MAX_NESTING )) || { echo "error: max nesting depth exceeded" >&2; exit 1; }

    # Walk `path` left-to-right, checking root/prefix for a physical archive
    local prefix="" inner="" archive="" rest="$path"

    while [[ -n "$rest" ]]; do
        local seg="${rest%%/*}"
        prefix="${prefix}${seg}"
        local candidate="$root/$prefix"

        if [[ "$rest" == "$seg" ]]; then
            rest=""
        else
            rest="${rest#*/}"
        fi

        if [[ -f "$candidate" ]]; then
            local fmt
            fmt=$(detect_format "$candidate" 2>/dev/null) || fmt=""
            if [[ -n "$fmt" ]]; then
                archive="$candidate"
                if [[ "$archive" != /* ]]; then
                    archive="$(cd "$(dirname "$archive")" 2>/dev/null && pwd)/$(basename "$archive")"
                fi
                inner="$rest"
                break
            fi
        fi

        prefix="${prefix}/"
    done

    # No archive found anywhere in this path — leaf case
    if [[ -z "$archive" ]]; then
        if (( depth == 0 )); then
            vim_call "$root/$path"; exit $?
        fi

        local target="$root/$path"

        if [[ -d "$target" ]]; then
            vim_call "$target" && _VIM_RC=0 || _VIM_RC=$?
            return 0
        fi

        [[ -f "$target" ]] || { echo "error: not found: $path" >&2; exit 1; }

        vim_call "$target" && _VIM_RC=0 || _VIM_RC=$?
        return 0
    fi

    # Archive (or compressed file) found at this level — extract and recurse
    validate_archive "$archive"
    local format
    format=$(detect_format "$archive")

    local tmpdir
    tmpdir=$(mktemp -d /tmp/vima.XXXXXX)
    TEMPDIRS+=("$tmpdir")

    extract_all "$archive" "$tmpdir" "$format"

    if [[ "$format" == gz || "$format" == bz2 || "$format" == xz ]]; then
        local decompressed="$tmpdir/$(stem "$archive" "$format")"
        [[ -f "$decompressed" ]] || { echo "error: decompressed file not found" >&2; exit 1; }

        local inner_fmt st
        inner_fmt=$(detect_format "$decompressed" 2>/dev/null) || inner_fmt=""
        if [[ -n "$inner_fmt" ]]; then
            st=$(stem "$archive" "$format")
            if [[ -z "$inner" || "$inner" == "$st" ]]; then
                walk "$tmpdir" "$st" "$((depth + 1))"
            elif [[ "$inner" == "$st"/* ]]; then
                walk "$tmpdir" "$inner" "$((depth + 1))"
            else
                walk "$tmpdir" "$st/$inner" "$((depth + 1))"
            fi
        else
            [[ -z "$inner" ]] || { echo "error: not an archive: $archive" >&2; exit 1; }
            vim_call "$decompressed" && _VIM_RC=0 || _VIM_RC=$?
        fi
    else
        walk "$tmpdir" "$inner" "$((depth + 1))"
    fi

    echo "updating $(basename "$archive")..."
    write_back "$archive" "$tmpdir" "$format"
    echo "done"
}

VIM_PRE=()
VIM_POST=()
INPUT=""
SEEN_DD=0
for arg in "$@"; do
    case "$arg" in
        --) SEEN_DD=1; continue ;;
    esac
    if (( SEEN_DD )); then
        if [[ -z "$INPUT" ]]; then INPUT="$arg"; else VIM_POST+=("$arg"); fi
    elif [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        usage; exit 0
    else
        VIM_PRE+=("$arg")
    fi
done
[[ -n "$INPUT" ]] || [[ ${#VIM_PRE[@]} -eq 0 ]] || {
    new_pre=() new_post=() found=0
    for a in "${VIM_PRE[@]:-}"; do
        if (( !found )) && [[ "$a" != -* && "$a" != +* ]]; then
            INPUT="$a"; found=1
        elif (( found )); then
            new_post+=("$a")
        else
            new_pre+=("$a")
        fi
    done
    if (( ${#new_pre[@]} )); then VIM_PRE=("${new_pre[@]}"); else VIM_PRE=(); fi
    if (( ${#new_post[@]} )); then VIM_POST=("${new_post[@]}"); else VIM_POST=(); fi
}
[[ -z "$INPUT" ]] && [[ ${#VIM_PRE[@]} -eq 0 && ${#VIM_POST[@]} -eq 0 ]] && { usage >&2; exit 1; }
[[ -n "$INPUT" ]] || { vim_call; exit $?; }

# Normalise leading ./ if present
[[ "$INPUT" == ./* ]] && INPUT="${INPUT#./}"

ensure_tool vim
ensure_tool file

walk "." "$INPUT" 0
exit $_VIM_RC
