#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s <destination-dir> [--force]\n' "$(basename "$0")" >&2
  printf 'Copies the bundled ecdsafail island GPU toolkit into a working directory.\n' >&2
}

dest=""
force=0

for arg in "$@"; do
  case "$arg" in
    --force)
      force=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ -n "$dest" ]; then
        usage
        exit 2
      fi
      dest="$arg"
      ;;
  esac
done

if [ -z "$dest" ]; then
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skill_dir="$(cd "$script_dir/.." && pwd)"
src="$skill_dir/assets/toolkit"

if [ ! -d "$src" ]; then
  printf 'Bundled toolkit not found at %s\n' "$src" >&2
  exit 1
fi

if [ -d "$dest" ] && [ "$force" -ne 1 ] && [ "$(find "$dest" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" != "0" ]; then
  printf 'Destination is not empty: %s\n' "$dest" >&2
  printf 'Use --force to copy over existing files without deleting extras.\n' >&2
  exit 3
fi

mkdir -p "$dest"

if command -v rsync >/dev/null 2>&1; then
  rsync -a "$src"/ "$dest"/
else
  cp -R "$src"/. "$dest"/
fi

chmod +x "$dest/island.sh" "$dest/runtime/"*.sh 2>/dev/null || true
printf 'Installed ecdsafail island GPU toolkit into %s\n' "$dest"
