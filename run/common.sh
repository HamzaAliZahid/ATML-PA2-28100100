# Shared helpers for the run scripts. Each step is skipped if it already finished
# (marker in results/.done/), so a crashed or restarted script resumes where it stopped.
set -u
mkdir -p results/.done logs
step() {
  local name="$1"; shift
  if [ -f "results/.done/$name" ]; then echo "[skip] $name (already done)"; return 0; fi
  echo "[$(date '+%H:%M:%S')] START $name :: $*"
  "$@"; local rc=$?
  if [ $rc -eq 0 ]; then
    touch "results/.done/$name"; echo "[$(date '+%H:%M:%S')] DONE  $name"
  else
    echo "[$(date '+%H:%M:%S')] FAILED $name (exit $rc) -- fix and rerun this script"
    if [ "${KEEP_GOING:-0}" = "1" ]; then return 0; fi   # unattended mode: continue with later steps
    exit 1
  fi
}
wait_for() {  # wait until a file exists (produced by the other GPU's script); give up after 3 h
  local n=0
  until [ -f "$1" ]; do
    echo "waiting for $1 ..."; sleep 60; n=$((n+1))
    if [ $n -ge 180 ]; then echo "gave up waiting for $1"; return 1; fi
  done
}
