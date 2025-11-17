# use to run distributed eval with 4 gpus on single node

CONFIG=scripts/eval/configs/h1_cma_cfg.py

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
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
EVAL_LOG="logs/${CONFIG_PREFIX}_eval.log"

torchrun \
  --nproc_per_node=8 \
  --master_port=2333 \
  scripts/eval/eval.py \
    --config $CONFIG \
  > $EVAL_LOG 2>&1
