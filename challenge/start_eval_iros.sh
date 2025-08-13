#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate internutopia

source /root/miniconda3/etc/profile.d/conda.sh
conda activate internutopia

ray disable-usage-stats
ray stop
ray start --head

CONFIG=scripts/eval/configs/challenge_cfg.py
SPLIT=""

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

# 从config文件名提取前缀
CONFIG_BASENAME=$(basename "$CONFIG" .py)
CONFIG_PREFIX=$(echo "$CONFIG_BASENAME" | sed 's/_cfg$//')

# 创建logs目录（如果不存在）
mkdir -p logs

# 设置日志文件路径
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


START_COMMAND_KUJIALE="python -u scripts/eval/eval_iros.py --config $CONFIG --default_config scripts/eval/configs/challenge_kujiale_cfg.py --split $SPLIT"
START_COMMAND_MP3D="python -u scripts/eval/eval_iros.py --config $CONFIG --default_config scripts/eval/configs/challenge_mp3d_cfg.py --split $SPLIT"
LOG_FILE="$EVAL_LOG"


rm eval_stdout.log
rm eval_stderr.log

start_process() {
    echo "Starting process..."
    # 连续跑两次，使用不同的dataset
    $START_COMMAND_MP3D > >(ansi2txt >> eval_stdout.log) 2> >(ansi2txt >> eval_stderr.log)
    $START_COMMAND_KUJIALE > >(ansi2txt >> eval_stdout.log) 2> >(ansi2txt >> eval_stderr.log)
}

start_process
