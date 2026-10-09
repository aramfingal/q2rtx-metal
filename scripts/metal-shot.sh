#!/bin/sh
# Runs the Metal renderer unattended and collects in-game screenshots (macOS).
#
#   scripts/metal-shot.sh OUTDIR COMMAND...
#
# Each COMMAND is one console line; "quit" is appended. For example:
#
#   scripts/metal-shot.sh /tmp/shots "map base1" "wait 200" "screenshot"
#
# The game runs with vid_hidden 1: no window, no Dock icon, no focus change, frames are
# drawn off-screen, so the machine stays usable. "wait N" waits N frames at 60 fps. Every
# run uses a throwaway home directory, so the user's config and saves are not touched.
# Screenshots land in OUTDIR as shot-N.png next to console.log. Set METAL_SHOT_TIMEOUT
# (seconds, default 120) for long runs.

set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 OUTDIR COMMAND..." >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
out=$1
shift
mkdir -p "$out" || exit 1
out=$(cd "$out" && pwd)

home=$(mktemp -d "${TMPDIR:-/tmp}/metal-shot.XXXXXX") || exit 1
trap 'rm -rf "$home"' EXIT
mkdir -p "$home/baseq2"
{
    printf '%s\n' "$@"
    printf 'wait 5\nquit\n'
} > "$home/baseq2/metal-shot.cfg"

rm -f "$out"/shot-*.png "$out/console.log"

cd "$root" || exit 1
SDL_MAC_BACKGROUND_APP=1 ./q2rtx +set vid_hidden 1 +set homedir "$home" \
    +set logfile 2 +set logfile_flush 1 +set s_enable 0 \
    +exec metal-shot.cfg > "$out/stdout.txt" 2>&1 &
pid=$!

status=0
elapsed=0
while kill -0 $pid 2>/dev/null; do
    if [ $elapsed -ge "${METAL_SHOT_TIMEOUT:-120}" ]; then
        echo "timed out" >&2
        kill -9 $pid
        status=1
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
wait $pid || status=1

cp "$home/baseq2/logs/console.log" "$out/console.log" 2>/dev/null
n=0
for f in "$home"/baseq2/screenshots/*.png; do
    [ -f "$f" ] || continue
    cp "$f" "$out/shot-$n.png"
    n=$((n + 1))
done
echo "$n screenshot(s) in $out"

exit $status
