#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

# Hugging Face token
TOKEN="${HF_TOKEN:-}"

# 服务器基础路径
SERVER_BASE_PATH="/mnt/data/kw/hy/projects/course/DL/hw2"

# =======================================================
# 1. 模型配置
# =======================================================
# 模型列表
MODELS=(
    "microsoft/resnet-18"                    # Task 1: CIFAR-10分类
    "facebook/detr-resnet-50"                # Task 2: 目标检测 (可选DETR)
    "nvidia/segformer-b0-finetuned-ade-512-512"  # Task 3: 语义分割参考
)

# 模型基础下载路径
BASE_MODEL_PATH="${SERVER_BASE_PATH}/models"

# =======================================================
# 2. 数据集配置
# =======================================================
# 数据集列表
DATASETS=(
    "cifar10"                                # Task 1: CIFAR-10
    "detection-datasets/coco"                # Task 2: COCO
)

# 数据集基础下载路径
BASE_DATASET_PATH="${SERVER_BASE_PATH}/datasets"

# =======================================================
# 3. 下载模型的逻辑
# =======================================================
# echo "====================================================="
# echo "             开始下载模型"
# echo "====================================================="

# # 循环遍历模型列表
# for repo_id in "${MODELS[@]}"; do
#     # 从仓库全名中提取模型名作为文件夹名
#     model_folder_name=$(basename "$repo_id")

#     # 拼接成完整的本地存储路径
#     local_repo_path="${BASE_MODEL_PATH}/${model_folder_name}"

#     echo "-----------------------------------------------------"
#     echo "模型仓库: $repo_id"
#     echo "将要保存到: $local_repo_path"
#     echo "-----------------------------------------------------"

#     # 确保目标文件夹存在
#     mkdir -p "$local_repo_path"

#     # 使用 until 循环处理下载失败的情况
#     max_retries=5
#     retry_count=0
#     while true; do
#         if huggingface-cli download \
#                 --token "$TOKEN" \
#                 --resume-download \
#                 --local-dir-use-symlinks False \
#                 "$repo_id" \
#                 --local-dir "$local_repo_path"; then
#             echo "模型 $repo_id 下载完成。"
#             break
#         fi

#         retry_count=$((retry_count + 1))
#         if [ "$retry_count" -ge "$max_retries" ]; then
#             echo "模型 $repo_id 已重试 ${max_retries} 次，跳过该模型。"
#             break
#         fi

#         echo "下载模型 $repo_id 失败，等待60s后重试（第 ${retry_count}/${max_retries} 次）..."
#         sleep 60
#     done

#     echo ""
# done

# # =======================================================
# # 4. 下载数据集的逻辑
# # =======================================================
# echo "====================================================="
# echo "             开始下载数据集"
# echo "====================================================="

# # 循环遍历数据集列表
# for repo_id in "${DATASETS[@]}"; do
#     # 从仓库全名中提取数据集名作为文件夹名
#     dataset_folder_name=$(basename "$repo_id")

#     # 拼接成完整的本地存储路径
#     local_repo_path="${BASE_DATASET_PATH}/${dataset_folder_name}"

#     echo "-----------------------------------------------------"
#     echo "数据集仓库: $repo_id"
#     echo "将要保存到: $local_repo_path"
#     echo "-----------------------------------------------------"

#     # 确保目标文件夹存在
#     mkdir -p "$local_repo_path"

#     # 使用重试循环处理下载失败的情况
#     max_retries=5
#     retry_count=0
#     while true; do
#         if [ -n "$TOKEN" ]; then
#             download_output=$(huggingface-cli download \
#                 --token "$TOKEN" \
#                 --repo-type dataset \
#                 --resume-download \
#                 --local-dir-use-symlinks False \
#                 "$repo_id" \
#                 --local-dir "$local_repo_path" 2>&1)
#         else
#             download_output=$(huggingface-cli download \
#                 --repo-type dataset \
#                 --resume-download \
#                 --local-dir-use-symlinks False \
#                 "$repo_id" \
#                 --local-dir "$local_repo_path" 2>&1)
#         fi
#         exit_code=$?

#         if [ "$exit_code" -eq 0 ]; then
#             echo "数据集 $repo_id 下载完成。"
#             break
#         fi

#         echo "$download_output"

#         if echo "$download_output" | grep -qiE "GatedRepoError|403 Client Error|restricted and you are not in the authorized list"; then
#             echo "数据集 $repo_id 访问受限（gated/403），跳过该数据集。"
#             break
#         fi

#         retry_count=$((retry_count + 1))
#         if [ "$retry_count" -ge "$max_retries" ]; then
#             echo "数据集 $repo_id 已重试 ${max_retries} 次，跳过该数据集。"
#             break
#         fi

#         echo "下载数据集 $repo_id 失败，60s后重试（第 ${retry_count}/${max_retries} 次）..."
#         sleep 60
#     done

#     echo ""
# done

echo ""
echo "====================================================="
echo "             下载 Stanford Background Dataset"
echo "====================================================="

STANFORD_DIR="${BASE_DATASET_PATH}/stanford_background"

if [ ! -d "$STANFORD_DIR" ]; then
    echo "正在下载 Stanford Background Dataset..."

    # 创建临时目录
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"

    # 下载数据集
    wget http://dags.stanford.edu/data/iccv09Data.tar.gz

    # 解压
    tar -xzf iccv09Data.tar.gz

    # 创建目标目录结构
    mkdir -p "$STANFORD_DIR/images"
    mkdir -p "$STANFORD_DIR/labels"

    # 移动图像文件
    echo "移动图像文件..."
    if [ -d "./iccv09Data/images" ]; then
        mv ./iccv09Data/images/*.jpg "$STANFORD_DIR/images/" 2>/dev/null || true
    fi

    # 移动标签文件（所有 .txt 文件）
    echo "移动标签文件..."
    if [ -d "./iccv09Data/labels" ]; then
        mv ./iccv09Data/labels/*.txt "$STANFORD_DIR/labels/" 2>/dev/null || true
    fi

    # 创建训练/测试划分
    cd "$STANFORD_DIR/images"
    if [ $(ls *.jpg 2>/dev/null | wc -l) -gt 0 ]; then
        ls *.jpg | head -600 > ../train.txt
        ls *.jpg | tail -115 > ../test.txt
    else
        echo "错误: 没有找到图像文件"
    fi

    # 清理临时文件
    rm -rf "$TMP_DIR"

    echo "✓ Stanford Background Dataset 下载完成"
    echo "  - 图像数量: $(ls $STANFORD_DIR/images/*.jpg 2>/dev/null | wc -l)"
    echo "  - 标签数量: $(ls $STANFORD_DIR/labels/*.png 2>/dev/null | wc -l)"
    echo "  - 训练集: $(wc -l < $STANFORD_DIR/train.txt)"
    echo "  - 测试集: $(wc -l < $STANFORD_DIR/test.txt)"
else
    echo "✓ Stanford Background Dataset 已存在"
fi

echo ""
echo "====================================================="
echo "所有下载任务已完成！"
echo "====================================================="
echo "数据集位置："
echo "  - CIFAR-10: ${BASE_DATASET_PATH}/cifar10/"
echo "  - COCO: ${BASE_DATASET_PATH}/coco/"
echo "  - Stanford Background: ${BASE_DATASET_PATH}/stanford_background/"
echo "  - Cityscapes: ${BASE_DATASET_PATH}/cityscapes/ (需手动下载)"
echo "====================================================="
