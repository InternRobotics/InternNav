
RETRY_LIMIT=5
MONITOR_INTERVAL=60
DEADLOCK_THRESHOLD=$((5 * 60))
pid=0
retry_count=0
LOG_FILE="eval_stderr.log"

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
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

run() {
    echo "Starting process..."
    pwd
    bash challenge/start_eval_iros.sh --split $SPLIT &
    pid=$!
}

run

sleep 1200 # wait for the process to start

while true; do
    sleep $MONITOR_INTERVAL
    echo "start healthcheck"

    if ! check_process; then
        if [ $retry_count -lt $RETRY_LIMIT ]; then
            echo "Retrying... (Attempt $((retry_count + 1))/$RETRY_LIMIT)"
            retry_count=$((retry_count + 1))
            run
        else
            echo "Exceeded maximum retry attempts. Exiting."
            exit 1
        fi
    else
        if ! check_log_update; then
            if [ $retry_count -lt $RETRY_LIMIT ]; then
                echo "Restarting process due to log file not updating... (Attempt $((retry_count + 1))/$RETRY_LIMIT)"
                retry_count=$((retry_count + 1))
                kill -9 $pid
                run
            else
                echo "Exceeded maximum retry attempts. Exiting."
                exit 1
            fi
        fi
    fi
done
