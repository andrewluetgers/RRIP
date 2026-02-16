#!/usr/bin/env bash
# resource-monitor.sh — Background resource monitoring for GPU encode profiling.
#
# Usage:
#   ./resource-monitor.sh <output_dir> start   # Start monitoring (writes PID file)
#   ./resource-monitor.sh <output_dir> stop    # Stop monitoring
#
# Captures:
#   - nvidia-smi dmon (GPU util%, mem util%, VRAM, power, temp) at 1s intervals
#   - CPU and RAM usage via vmstat/top at 1s intervals
#
# Output files in <output_dir>:
#   - gpu_monitor.csv      — nvidia-smi dmon output
#   - cpu_monitor.csv      — CPU/RAM usage
#   - monitor.pid          — PID file for cleanup

set -euo pipefail

OUTPUT_DIR="${1:?Usage: resource-monitor.sh <output_dir> start|stop}"
ACTION="${2:?Usage: resource-monitor.sh <output_dir> start|stop}"

PID_FILE="$OUTPUT_DIR/monitor.pid"

start_monitoring() {
    mkdir -p "$OUTPUT_DIR"

    # GPU monitoring via nvidia-smi dmon
    # Columns: gpu sm mem enc dec mclk pclk pviol tw fb bar1 sbecc dbecc pci rxpci txpci
    # We use -d 1 for 1-second intervals
    (
        echo "# timestamp,gpu_idx,sm_util%,mem_util%,fb_used_mb,power_w,temp_c"
        nvidia-smi dmon -d 1 -s pum 2>/dev/null | while read -r line; do
            # Skip header lines (start with #)
            [[ "$line" =~ ^# ]] && continue
            [[ -z "$line" ]] && continue
            ts=$(date +%s.%N)
            echo "$ts,$line"
        done
    ) > "$OUTPUT_DIR/gpu_monitor.csv" 2>/dev/null &
    GPU_PID=$!

    # CPU/RAM monitoring
    (
        echo "# timestamp,cpu_user%,cpu_sys%,cpu_idle%,mem_total_kb,mem_used_kb,mem_free_kb"
        while true; do
            ts=$(date +%s.%N)
            # Get memory info
            if command -v free &>/dev/null; then
                mem_line=$(free -k | grep '^Mem:')
                mem_total=$(echo "$mem_line" | awk '{print $2}')
                mem_used=$(echo "$mem_line" | awk '{print $3}')
                mem_free=$(echo "$mem_line" | awk '{print $4}')
            else
                mem_total=0; mem_used=0; mem_free=0
            fi
            # Get CPU from /proc/stat (Linux) or top (fallback)
            if [[ -f /proc/stat ]]; then
                read -r _ user nice system idle rest < /proc/stat 2>/dev/null <<< "$(head -1 /proc/stat)"
                total=$((user + nice + system + idle))
                if [[ -n "${prev_total:-}" ]]; then
                    d_total=$((total - prev_total))
                    d_idle=$((idle - prev_idle))
                    if [[ $d_total -gt 0 ]]; then
                        cpu_busy=$(( (d_total - d_idle) * 100 / d_total ))
                        cpu_idle_pct=$(( d_idle * 100 / d_total ))
                    else
                        cpu_busy=0; cpu_idle_pct=100
                    fi
                    echo "$ts,$cpu_busy,0,$cpu_idle_pct,$mem_total,$mem_used,$mem_free"
                fi
                prev_total=$total
                prev_idle=$idle
            fi
            sleep 1
        done
    ) > "$OUTPUT_DIR/cpu_monitor.csv" 2>/dev/null &
    CPU_PID=$!

    echo "$GPU_PID $CPU_PID" > "$PID_FILE"
    echo "Monitoring started (PIDs: $GPU_PID $CPU_PID) -> $OUTPUT_DIR"
}

stop_monitoring() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No monitor.pid found in $OUTPUT_DIR"
        return 0
    fi
    read -r GPU_PID CPU_PID < "$PID_FILE"
    kill "$GPU_PID" 2>/dev/null || true
    kill "$CPU_PID" 2>/dev/null || true
    rm -f "$PID_FILE"

    # Print summary
    if [[ -f "$OUTPUT_DIR/gpu_monitor.csv" ]]; then
        local gpu_lines
        gpu_lines=$(grep -v '^#' "$OUTPUT_DIR/gpu_monitor.csv" | wc -l)
        echo "GPU monitor: $gpu_lines samples captured"
    fi
    if [[ -f "$OUTPUT_DIR/cpu_monitor.csv" ]]; then
        local cpu_lines
        cpu_lines=$(grep -v '^#' "$OUTPUT_DIR/cpu_monitor.csv" | wc -l)
        echo "CPU monitor: $cpu_lines samples captured"
    fi
    echo "Monitoring stopped."
}

case "$ACTION" in
    start) start_monitoring ;;
    stop)  stop_monitoring ;;
    *)     echo "Unknown action: $ACTION (use start or stop)"; exit 1 ;;
esac
