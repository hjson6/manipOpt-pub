#!/usr/bin/env bash
# Shared by the scenario scripts; source it and call
#   run_scenario "<package>" "<file>.launch.py" [launch args...]
# It kills leftovers, launches, waits for mpc_controller's ready line,
# prompts, and sends /mpc/go.
#
# Leftovers are killed first because the viewer can segfault on Ctrl+C and
# leave a process stuck that keeps its DDS participant, so the next launch
# ran two sets of nodes. SIGKILL in turn leaves FastDDS shared-memory
# segments in /dev/shm, so unused ones (nothing has them open) are removed.
run_scenario() {
    local launch_pkg="$1"
    local launch_file="$2"
    shift 2  # the rest are launch arguments
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    pkill -9 -f "lib/pick_place_" 2>/dev/null || true
    pkill -9 -f "ros2 run pick_place_" 2>/dev/null || true
    pkill -9 -f "ros2 launch pick_place_" 2>/dev/null || true
    sleep 1

    for f in /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; do
        [ -e "$f" ] || continue
        lsof -t "$f" >/dev/null 2>&1 || rm -f "$f"
    done

    source "$repo_root/scripts/env.sh"

    # Log to a file (scanned for the ready line) and tail it, rather than
    # piping into tee, so that $! is ros2 launch's PID (it gets the SIGINT).
    local log
    log="$(mktemp)"
    local launch_pid tail_pid
    cleanup() {
        kill -INT "$launch_pid" 2>/dev/null || true
        wait "$launch_pid" 2>/dev/null || true
        kill "$tail_pid" 2>/dev/null || true
        rm -f "$log"
    }
    trap cleanup EXIT INT TERM

    ros2 launch "$launch_pkg" "$launch_file" "$@" > "$log" 2>&1 &
    launch_pid=$!
    tail -n +1 -f "$log" &
    tail_pid=$!

    # Must match mpc_controller_node.py's READY_LOG_MESSAGE.
    until grep -q "mpc_controller ready; waiting for /mpc/go to start" "$log" 2>/dev/null; do
        if ! kill -0 "$launch_pid" 2>/dev/null; then
            echo "Pipeline exited before becoming ready -- see the log above." >&2
            exit 1
        fi
        sleep 0.5
    done

    if [ -n "$AUTO_START" ]; then
        sleep 5  # the rest of the stack (the base's, on the mobile job) up too
    else
        sleep 5  # the rest of the stack up too, so its start-up lines come before the prompt
        # The log stops scrolling while it waits, so the prompt stays in view; keys typed
        # during start-up are dropped, so a stray ENTER cannot start it.
        kill "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true
        if [ -t 0 ]; then
            while read -r -t 0.1 -n 10000 _; do :; done
        fi
        read -rp $'\n>>> Pipeline is up and holding (the log resumes once started). Press ENTER to start operating... '
        tail -n 0 -f "$log" &
        tail_pid=$!
    fi

    # One message can go out before the controller is discovered: until it says it started.
    local tries=0
    until grep -q "received /mpc/go; starting operation" "$log"; do
        ros2 topic pub --once /mpc/go std_msgs/msg/Empty '{}' > /dev/null
        tries=$((tries + 1))
        [ "$tries" -ge 15 ] && { echo "the controller never started -- see the log above." >&2; break; }
        sleep 2
    done

    wait "$launch_pid"
}
