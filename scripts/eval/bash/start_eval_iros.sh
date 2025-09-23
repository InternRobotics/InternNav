#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate internutopia

source /root/miniconda3/etc/profile.d/conda.sh
conda activate internutopia

CONFIG=scripts/eval/configs/challenge_cfg.py
SPLIT=""

export CUDA_VISIBLE_DEVICES=0,1

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Extract the prefix from the config filename
CONFIG_BASENAME=$(basename "$CONFIG" .py)
CONFIG_PREFIX=$(echo "$CONFIG_BASENAME" | sed 's/_cfg$//')

# Create the logs directory if it doesn't exist
mkdir -p logs

# Set the log file paths
SERVER_LOG="logs/${CONFIG_PREFIX}_server.log"
EVAL_LOG="logs/${CONFIG_PREFIX}_eval.log"

processes=$(ps -ef | grep 'internnav/agent/utils/server.py' | grep -v grep | awk '{print $2}')
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 $pid
        echo "kill: $pid"
    done
fi
python internnav/agent/utils/server.py --config scripts/eval/configs/challenge_kujiale_cfg.py > "$SERVER_LOG" 2>&1 &


RETRY_LIMIT=5
MONITOR_INTERVAL=60
DEADLOCK_THRESHOLD=$((5 * 60))

START_COMMAND_KUJIALE="python -u scripts/eval/eval_iros.py --config $CONFIG --default_config scripts/eval/configs/challenge_kujiale_cfg.py --split $SPLIT"
START_COMMAND_MP3D="python -u scripts/eval/eval_iros.py --config $CONFIG --default_config scripts/eval/configs/challenge_mp3d_cfg.py --split $SPLIT"
LOG_FILE="$EVAL_LOG"

pid=0

retry_count=0

rm eval_stdout.log
rm eval_stderr.log

start_process() {
    echo "Starting process..."
    $START_COMMAND_MP3D > >(ansi2txt >> eval_stdout.log) 2> >(ansi2txt >> eval_stderr.log)
    $START_COMMAND_KUJIALE > >(ansi2txt >> eval_stdout.log) 2> >(ansi2txt >> eval_stderr.log)
}


check_process() {
    if ! kill -0 $pid > /dev/null 2>&1; then
        echo "Process $pid has exited."
        return 1
    fi
    return 0
}


check_log_update() {
    if [ ! -e "$LOG_FILE" ]; then
        return 1
    fi
    last_update=$(stat -c %Y "$LOG_FILE")
    current_time=$(date +%s)

    delta=$(( current_time - last_update ))

    if [ $delta -ge $DEADLOCK_THRESHOLD ]; then
        echo "Log file has not been updated for $((DEADLOCK_THRESHOLD / 60)) minutes."
        return 1
    fi

    return 0
}

start_process

# terminate server process
processes=$(ps -ef | grep 'internnav/agent/utils/server.py' | grep -v grep | awk '{print $2}')
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 $pid
        echo "kill: $pid"
    done
fi
