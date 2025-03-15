#!/bin/bash
# 测试训练好的DGPPO模型在MPEFoV环境中的表现

# 设置参数
NUM_AGENTS=8          # 智能体数量
NUM_EPISODES=5        # 测试轮数
SEED=42               # 测试随机种子
LOG_DIR="./logs"      # 日志目录
MAX_VIDEOS=2          # 生成视频的最大数量

echo "======================="
echo "开始测试 DGPPO 算法在 MPEFoV 环境中"
echo "======================="
echo "智能体数量: $NUM_AGENTS"
echo "测试轮数: $NUM_EPISODES"
echo "测试随机种子: $SEED"
echo "======================="

# 从日志目录中找到最新的训练目录
if [ -f "last_dgppo_train_dir.txt" ]; then
    MODEL_DIR=$(cat last_dgppo_train_dir.txt)
    echo "使用最近的训练目录: $MODEL_DIR"
else
    # 如果没有记录文件，则查找最新的训练目录
    LATEST_DIR=$(find $LOG_DIR/MPEFoV/dgppo -type d -name "seed*" | sort -r | head -n 1)
    if [ -z "$LATEST_DIR" ]; then
        echo "找不到训练目录，请先运行训练脚本"
        exit 1
    fi
    MODEL_DIR=$LATEST_DIR
    echo "使用最新的训练目录: $MODEL_DIR"
fi

# 运行测试
python test.py \
    --env MPEFoV \
    --algo dgppo \
    --num-agents $NUM_AGENTS \
    --obs 0 \
    --model-dir $MODEL_DIR \
    --epi $NUM_EPISODES \
    --seed $SEED \
    --stochastic \
    --max-videos $MAX_VIDEOS

# 检查测试是否成功
if [ $? -ne 0 ]; then
    echo "测试失败"
    exit 1
fi

echo "======================="
echo "测试完成"
echo "=======================" 