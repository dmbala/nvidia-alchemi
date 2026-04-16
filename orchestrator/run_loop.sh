#!/bin/bash
# run_loop.sh — drive the /ht-loop orchestrator from a tmux session.
#
# Commands:
#   start   create tmux session running the tick loop
#   status  show session state + orchestrator snapshot
#   attach  attach to the running session
#   stop    kill the session
#   tick    run one tick right now (bypasses tmux)
#   logs    tail the latest log file
#   help
#
# All defaults live in ALCHEMI_* env vars (see defaults below). The main
# reasons to override are session name (to run multiple loops) and state
# paths (to loop against a different run directory).

set -euo pipefail

# --- configuration -------------------------------------------------------

SESSION="${ALCHEMI_SESSION:-alchemi-loop}"
INTERVAL="${ALCHEMI_INTERVAL:-900}"                              # seconds between ticks
DRIVER="${ALCHEMI_DRIVER:-bash}"                                 # bash | agent
MODE="${ALCHEMI_MODE:-research}"                                 # screen | research
STATE="${ALCHEMI_STATE:-runs/prod/screen_state.json}"
AL_STATE="${ALCHEMI_AL_STATE:-runs/prod/al_state.json}"
AL_ROOT="${ALCHEMI_AL_ROOT:-runs/prod/al}"
CHUNK_DIR="${ALCHEMI_CHUNK_DIR:-runs/prod/chunks}"
RESULT_DIR="${ALCHEMI_RESULT_DIR:-runs/prod/results}"
SLURM_SCRIPT="${ALCHEMI_SLURM_SCRIPT:-alchemi_ht/run_screening.slurm}"
UNEXPLORED="${ALCHEMI_UNEXPLORED:-runs/prod/subset.csv}"
MIN_RESULTS="${ALCHEMI_MIN_RESULTS:-1}"
LOG_DIR="${ALCHEMI_LOG_DIR:-runs/prod/logs}"
PYTHON="${ALCHEMI_PYTHON:-python3.12}"
CLAUDE_BIN="${ALCHEMI_CLAUDE:-claude}"
CLAUDE_MODEL="${ALCHEMI_CLAUDE_MODEL:-sonnet}"
CLAUDE_PERM="${ALCHEMI_CLAUDE_PERM:-auto}"                       # auto | acceptEdits | bypassPermissions | default
CLAUDE_MAX_TURNS="${ALCHEMI_CLAUDE_MAX_TURNS:-8}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- argument builders ---------------------------------------------------

build_tick_args() {
    local args=(
        --mode "$MODE"
        --action tick
        --state "$STATE"
        --chunk-dir "$CHUNK_DIR"
        --result-dir "$RESULT_DIR"
        --slurm-script "$SLURM_SCRIPT"
        --min-results "$MIN_RESULTS"
    )
    if [ "$MODE" = "research" ]; then
        args+=(
            --al-state "$AL_STATE"
            --al-root "$AL_ROOT"
            --unexplored-csv "$UNEXPLORED"
        )
    fi
    printf '%s\n' "${args[@]}"
}

build_status_args() {
    local args=(
        --mode "$MODE"
        --action status
        --state "$STATE"
    )
    if [ "$MODE" = "research" ]; then
        args+=(--al-state "$AL_STATE")
    fi
    printf '%s\n' "${args[@]}"
}

# --- commands ------------------------------------------------------------

build_agent_prompt() {
    # Printed once into the inner shell, quoted, passed as a single argv to claude -p.
    cat <<EOF
You are the autonomous driver for the ALCHEMI /ht-loop orchestrator. Each invocation of yours is one tick.

Per tick:
1. Run the tick command shown below. This submits/polls SLURM, advances phases, updates state files.
2. Read the tick's stdout. If nothing changed since the prior tick, just say "no change" and stop.
3. Inspect pending_decisions in the state files:
   - runs/prod/screen_state.json
   - runs/prod/al_state.json
4. Handle decisions by *kind*:
   - "anomalies" (sacct COMPLETED but no CSV): do NOT retry. Summarize the affected chunks and which SLURM error log to inspect (logs/screen_<job>_<task>.err). Leave in pending_decisions.
   - "permanent_failures" (3-retry cap): already skipped by the orchestrator. Note them and continue.
   - "al_phase_failed": critical. Read the associated SLURM .err log, identify the failure class (OOM / timeout / code bug), summarize in one sentence. Do NOT restart the AL phase — flag for human.
5. Report in ≤4 lines: phase changes, new pending decisions, and one next-step suggestion if any.

Tick command:
  ${TICK_CMD_STR}

Constraints:
- You may run the tick command, read files under runs/ and logs/, and run squeue/sacct read-only.
- Do NOT edit state files by hand. Do NOT scancel jobs. Do NOT modify code.
- Stay terse. The wrapper loop will re-invoke you in ${INTERVAL}s.
EOF
}

cmd_start() {
    command -v tmux >/dev/null || { echo "ERROR: tmux not found on PATH"; exit 1; }
    command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not found"; exit 1; }
    if [ "$DRIVER" = "agent" ]; then
        command -v "$CLAUDE_BIN" >/dev/null || { echo "ERROR: '$CLAUDE_BIN' not found (needed for DRIVER=agent)"; exit 1; }
    elif [ "$DRIVER" != "bash" ]; then
        echo "ERROR: DRIVER must be 'bash' or 'agent' (got: $DRIVER)"; exit 1
    fi

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' already running."
        echo "  Attach:  bash $0 attach"
        echo "  Stop:    bash $0 stop"
        exit 0
    fi
    if [ ! -f "$STATE" ]; then
        echo "ERROR: screen state missing at $STATE"
        echo "Initialize first:"
        echo "  $PYTHON -m orchestrator.ht_loop --mode screen --action init ..."
        exit 2
    fi
    if [ "$MODE" = "research" ] && [ ! -f "$AL_STATE" ]; then
        echo "ERROR: AL state missing at $AL_STATE (required for --mode research)"
        echo "Initialize with: $PYTHON -m orchestrator.ht_loop --mode research --action init ..."
        exit 2
    fi

    mkdir -p "$LOG_DIR"
    local ts log tick_cmd inner_cmd loop_cmd
    ts="$(date +%Y%m%d_%H%M%S)"
    log="$LOG_DIR/loop_${ts}_${DRIVER}.log"

    # Build the tick command as a single properly-quoted string.
    tick_cmd=$(printf '%q ' "$PYTHON" -m orchestrator.ht_loop)
    while IFS= read -r a; do tick_cmd+=$(printf '%q ' "$a"); done < <(build_tick_args)
    TICK_CMD_STR="$tick_cmd"

    if [ "$DRIVER" = "bash" ]; then
        # Pure bash: shell runs tick directly.
        inner_cmd="$tick_cmd"
    else
        # Agent: each tick is a headless `claude -p` call. The agent runs the tick
        # itself and inspects state. We write the prompt to a sibling file so it
        # stays auditable and survives restarts.
        local prompt_file="$LOG_DIR/agent_prompt_${ts}.txt"
        build_agent_prompt > "$prompt_file"
        # Keep the cd inline so we're in PROJECT_ROOT when claude runs.
        inner_cmd=$(printf '%q --model %q --permission-mode %q --max-turns %q -p "$(cat %q)"' \
            "$CLAUDE_BIN" "$CLAUDE_MODEL" "$CLAUDE_PERM" "$CLAUDE_MAX_TURNS" "$prompt_file")
    fi

    # Inline loop: cd, log header, run inner_cmd, pause, repeat. Output tee'd to a file.
    loop_cmd="cd $(printf '%q' "$PROJECT_ROOT") && while true; do \
echo '=== tick '\"\$(date)\"' (driver=${DRIVER}) ==='; \
${inner_cmd} 2>&1; \
echo ''; \
sleep ${INTERVAL}; \
done 2>&1 | tee -a $(printf '%q' "$log")"

    # Pane 0 (main, left, ~60% width): the tick loop.
    tmux new-session -d -s "$SESSION" -n orchestrator -c "$PROJECT_ROOT"
    tmux send-keys -t "$SESSION:orchestrator.0" "$loop_cmd" C-m

    # Pane 1 (top-right): live squeue, --array so each task shows individually.
    tmux split-window -h -p 40 -t "$SESSION:orchestrator.0" -c "$PROJECT_ROOT"
    tmux send-keys -t "$SESSION:orchestrator.1" \
        'watch -n 10 "squeue -u \$USER --array --states=all 2>/dev/null | head -40"' C-m

    # Pane 2 (bottom-right): result/AL artifact counts.
    tmux split-window -v -p 50 -t "$SESSION:orchestrator.1" -c "$PROJECT_ROOT"
    tmux send-keys -t "$SESSION:orchestrator.2" \
        "watch -n 15 'echo -n \"screen results: \"; ls $RESULT_DIR 2>/dev/null | wc -l; echo; echo AL iters:; ls $AL_ROOT 2>/dev/null || echo none; echo; echo pending decisions:; python3.12 -c \"import json; s=json.load(open(\\\"$AL_STATE\\\")); print(len(s.get(\\\"pending_decisions\\\",[])))\" 2>/dev/null || true'" C-m

    echo "Session '$SESSION' started with 3 panes."
    echo "  Attach:     bash $0 attach        (Ctrl-b d to detach)"
    echo "  Live log:   tail -f $log"
    echo "  Status:     bash $0 status"
    echo "  Stop:       bash $0 stop"
}

cmd_status() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux: session '$SESSION' is running ($(tmux list-panes -t $SESSION -F '' | wc -l) panes)."
    else
        echo "tmux: session '$SESSION' not running."
    fi
    echo "---"
    cd "$PROJECT_ROOT"
    local args=()
    while IFS= read -r a; do args+=("$a"); done < <(build_status_args)
    "$PYTHON" -m orchestrator.ht_loop "${args[@]}"
}

cmd_attach() {
    tmux attach -t "$SESSION"
}

cmd_stop() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        echo "Stopped session '$SESSION'."
    else
        echo "No session '$SESSION' to stop."
    fi
}

cmd_tick() {
    cd "$PROJECT_ROOT"
    local args=()
    while IFS= read -r a; do args+=("$a"); done < <(build_tick_args)
    "$PYTHON" -m orchestrator.ht_loop "${args[@]}"
}

cmd_logs() {
    ls -lt "$LOG_DIR"/loop_*.log 2>/dev/null | head -5
    local latest
    latest=$(ls -t "$LOG_DIR"/loop_*.log 2>/dev/null | head -1)
    if [ -n "${latest:-}" ]; then
        echo "---"
        echo "Tailing $latest (Ctrl-C to stop)..."
        tail -f "$latest"
    else
        echo "No logs under $LOG_DIR yet."
    fi
}

cmd_help() {
    cat <<EOF
Usage: bash $0 {start|status|attach|stop|tick|logs|help}

Commands:
  start   Create a tmux session running a tick loop every ${INTERVAL}s.
          Panes: tick loop | squeue watch | result/AL-iter counts.
  status  Show tmux session state + current orchestrator summary.
  attach  Attach to the running session (Ctrl-b then d to detach).
  stop    Kill the session.
  tick    Run a single tick now without involving tmux.
  logs    Tail the most recent tick-loop log.

Environment variables (ALCHEMI_* prefix, current values):
  SESSION=$SESSION
  INTERVAL=$INTERVAL (seconds)
  DRIVER=$DRIVER  (bash | agent)
  MODE=$MODE  (screen | research)
  STATE=$STATE
  AL_STATE=$AL_STATE
  AL_ROOT=$AL_ROOT
  CHUNK_DIR=$CHUNK_DIR
  RESULT_DIR=$RESULT_DIR
  SLURM_SCRIPT=$SLURM_SCRIPT
  UNEXPLORED=$UNEXPLORED
  MIN_RESULTS=$MIN_RESULTS
  LOG_DIR=$LOG_DIR
  PYTHON=$PYTHON

Agent-mode extras (only when DRIVER=agent):
  CLAUDE=$CLAUDE_BIN
  CLAUDE_MODEL=$CLAUDE_MODEL
  CLAUDE_PERM=$CLAUDE_PERM  (auto | acceptEdits | bypassPermissions | default)
  CLAUDE_MAX_TURNS=$CLAUDE_MAX_TURNS

Drivers:
  bash   (default, \$0 runs the tick command directly each interval).
         Zero dependencies besides tmux + python. Deterministic, auditable,
         never reads pending_decisions.
  agent  Each interval invokes \`claude -p\` with a prompt telling it to run
         the tick AND inspect pending_decisions. Agent reads state, triages
         anomalies/failures, and summarizes in its own words. Doesn't touch
         state files or cancel jobs on its own.

Examples:
  # Agent-free (current default):
  bash $0 start

  # Agent-orchestrated (each tick is one claude invocation):
  ALCHEMI_DRIVER=agent bash $0 start

  # Second loop, different state, bash driver, faster cadence:
  ALCHEMI_SESSION=alchemi-dev ALCHEMI_STATE=runs/dev/screen_state.json \\
  ALCHEMI_INTERVAL=600 bash $0 start
EOF
}

# --- dispatch -----------------------------------------------------------

case "${1:-help}" in
    start)   cmd_start ;;
    status)  cmd_status ;;
    attach)  cmd_attach ;;
    stop)    cmd_stop ;;
    tick)    cmd_tick ;;
    logs)    cmd_logs ;;
    help|-h|--help) cmd_help ;;
    *)       echo "Unknown command: ${1}"; cmd_help; exit 1 ;;
esac
