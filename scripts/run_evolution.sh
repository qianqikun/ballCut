#!/bin/bash

# BallCut AI Evolution - Master Orchestrator Script
# This script guides you through exporting data and training your custom model.

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# Colors for better UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}       🏀 BallCut AI 模型自我进化工作流        ${NC}"
echo -e "${BLUE}================================================${NC}"

# 1. Environment Check
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 Python ($VENV_PYTHON)${NC}"
    exit 1
fi

# 2. Step 1: Export Data
echo -e "\n${YELLOW}步骤 1: 正在从您的审核记录中导出训练数据...${NC}"
$VENV_PYTHON "$SCRIPT_DIR/export_training_data.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}导出失败。请确保您已经通过 Web 界面审核并“确认”了一些进球。${NC}"
    exit 1
fi

# 3. Step 2: Training Configuration
echo -e "\n${YELLOW}步骤 2: 配置训练参数${NC}"
read -p "请输入训练轮数 (Epochs) [默认: 50]: " EPOCHS
EPOCHS=${EPOCHS:-50}

read -p "请输入批次大小 (Batch Size) [默认: 16]: " BATCH
BATCH=${BATCH:-16}

echo -e "\n${GREEN}即将开始训练...${NC}"
echo -e "参数: Epochs=$EPOCHS, Batch=$BATCH"
echo -e "提示: 系统将自动启用 Mac MPS (Apple Silicon) 加速。"

# 4. Step 3: Run Training
$VENV_PYTHON "$SCRIPT_DIR/train_model.py" --epochs "$EPOCHS" --batch "$BATCH"

if [ $? -ne 0 ]; then
    echo -e "${RED}训练过程中止。${NC}"
    exit 1
fi

# 5. Step 4: Deployment Options
echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}✅ 训练完成！${NC}"
echo -e "${BLUE}================================================${NC}"

NEW_MODEL="$PROJECT_ROOT/data/runs/ball_refinement/weights/best.pt"
TARGET_MODEL="$PROJECT_ROOT/models/best.pt"

if [ -f "$NEW_MODEL" ]; then
    echo -e "\n新模型已生成在: ${YELLOW}$NEW_MODEL${NC}"
    read -p "是否立即部署该模型（替换当前使用的 models/best.pt）? (y/n): " DEPLOY
    if [[ $DEPLOY == "y" || $DEPLOY == "Y" ]]; then
        # Backup old model
        cp "$TARGET_MODEL" "${TARGET_MODEL}.bak"
        # Deploy new model
        cp "$NEW_MODEL" "$TARGET_MODEL"
        echo -e "${GREEN}🚀 部署成功！旧模型已备份为 best.pt.bak。${NC}"
        echo -e "${GREEN}重启项目后，新模型将生效。${NC}"
    else
        echo -e "已跳过自动部署。您可以稍后手动替换模型。"
    fi
else
    echo -e "${RED}未能在预期路径找到新生成的模型文件。${NC}"
fi

echo -e "\n${BLUE}感谢使用 BallCut AI 进化工具！${NC}"
