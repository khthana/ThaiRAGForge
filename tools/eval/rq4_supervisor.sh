#!/usr/bin/env bash
# Supervised RQ4 refresh: run each (model, variant) to completion, restarting on
# a non-zero exit or a stall, until every arm reports 106 valid answers.
#
# Three things this adds over running rq4_generate.py by hand:
#   1. PURGE between attempts. A failed generation still writes an answer file
#      (answer="", error=<exc>) and an existing file is skipped on resume, so
#      without this a stalled cell is frozen as an empty answer and scored.
#   2. Completion is counted over answers with error=None, never over file
#      count -- an error-carrying file would otherwise make an arm look done.
#   3. A stall watchdog on the newest ANSWER FILE's mtime, not on the log:
#      the log can be legitimately quiet for ~7 min (progress prints every 10
#      answers, and the slowest observed answer is 405 s). Threshold 1800 s is
#      above one 900 s request timeout plus a 405 s slow answer (1305 s), so a
#      single timeout followed by a slow success cannot false-fire.
#
# ARMS is passed explicitly on purpose: the default is "every context dir",
# which would generate 212 entity-arm answers per variant that no published
# variant has ever carried.
set -u
cd /c/Users/Terry/Desktop/Code/RAG

SP="$(dirname "$0")"   # rq4_status.py sits beside this script
LOG=data/logs/rq4_supervised_2026_08_19.log
ARMS=hybrid_qwen3_0.6b_semantic,dense_qwen3_0.6b_semantic,bm25_semantic,hybrid_m2v_semantic
MAX_ATTEMPTS=6
STALL_S=11400
DRY=${DRY:-0}
PY=.venv/Scripts/python.exe

# model:variant:outdir
JOBS="
phi4:sentence_cap:phi4
phi4:cite_all:phi4_cite_all
phi4:cite_all_guarded:phi4_cite_all_guarded
gemma4:e4b|cite_all|gemma4_e4b_cite_all
gemma4:e4b|cite_all_guarded|gemma4_e4b_cite_all_guarded
"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

newest_answer_age() {  # seconds of silence: no new answer under $1 since $2 (attempt start)
  # $2 is load-bearing. Measuring from the newest EXISTING answer alone reports
  # the age of yesterday's frozen answers (~18 h) and trips the watchdog on the
  # first check of every attempt -- which is what happened on the 07:29 run.
  # Liveness is "nothing new since this attempt began", so the clock starts at
  # the later of (attempt start, newest answer).
  "$PY" - "$1" "$2" <<'PYEOF'
import os,sys,time
base=os.path.join("data/rq4/answers",sys.argv[1])
m=float(sys.argv[2])
for r,_,fs in os.walk(base):
    for f in fs:
        m=max(m,os.path.getmtime(os.path.join(r,f)))
print(int(time.time()-m))
PYEOF
}

kill_generate() {  # kill only the rq4_generate python, found by command line
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*rq4_generate*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
}

unload_all() {  # a killed run can leave the model resident; the next run refuses to start then
  for m in phi4 gemma4:e4b; do ollama stop "$m" >/dev/null 2>&1; done
  sleep 3
}

say "=== supervisor start (DRY=$DRY) ==="
for job in $JOBS; do
  MODEL=$(echo "$job" | awk -F'[:|]' '{print $1}')
  case "$job" in
    *"|"*) MODEL=$(echo "$job" | cut -d'|' -f1)
           VARIANT=$(echo "$job" | cut -d'|' -f2)
           OUTDIR=$(echo "$job" | cut -d'|' -f3) ;;
    *)     MODEL=$(echo "$job" | cut -d: -f1)
           VARIANT=$(echo "$job" | cut -d: -f2)
           OUTDIR=$(echo "$job" | cut -d: -f3) ;;
  esac

  say "--- job $MODEL / $VARIANT -> answers/$OUTDIR ---"
  if "$PY" "$SP/rq4_status.py" "$OUTDIR" >>"$LOG" 2>&1; then
    say "already complete, skipping"
    continue
  fi

  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    "$PY" "$SP/rq4_status.py" "$OUTDIR" --purge 2>&1 | tee -a "$LOG"
    unload_all
    say "attempt $attempt/$MAX_ATTEMPTS: $MODEL $VARIANT"
    if [ "$DRY" = "1" ]; then
      echo "DRY: PYTHONPATH=src "$PY" -u tools/eval/rq4_generate.py --model $MODEL --variant $VARIANT --arms $ARMS" | tee -a "$LOG"
      break
    fi

    PYTHONPATH=src PYTHONIOENCODING=utf-8 "$PY" -u tools/eval/rq4_generate.py \
        --model "$MODEL" --variant "$VARIANT" --arms "$ARMS" >>"$LOG" 2>&1 &
    PID=$!
    T0=$(date +%s)

    while kill -0 $PID 2>/dev/null; do
      sleep 120
      AGE=$(newest_answer_age "$OUTDIR" "$T0")
      if [ "$AGE" -gt "$STALL_S" ]; then
        say "STALL: no new answer for ${AGE}s (> $STALL_S) -- killing and restarting"
        kill_generate; kill $PID 2>/dev/null; sleep 5
        break
      fi
    done
    wait $PID 2>/dev/null; RC=$?
    say "attempt $attempt finished rc=$RC"

    if "$PY" "$SP/rq4_status.py" "$OUTDIR" >>"$LOG" 2>&1; then
      say "COMPLETE: $OUTDIR"
      break
    fi
    say "incomplete, retrying in 60s"
    sleep 60
  done

  if ! "$PY" "$SP/rq4_status.py" "$OUTDIR" >>"$LOG" 2>&1; then
    say "!! GIVING UP on $OUTDIR after $MAX_ATTEMPTS attempts -- stopping"
    exit 1
  fi
done
say "=== supervisor done: all 5 variants complete ==="
