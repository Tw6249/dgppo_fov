#!/bin/bash
# 激活dgppo环境并运行FOV环境测试脚本

echo "正在设置dgppo环境..."

# 添加当前目录到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 如果存在Conda环境，可以取消下面几行的注释
# if command -v conda &> /dev/null; then
#     if conda env list | grep -q dgppo; then
#         echo "激活dgppo conda环境..."
#         source $(conda info --base)/etc/profile.d/conda.sh
#         conda activate dgppo
#     else
#         echo "警告: dgppo conda环境不存在"
#     fi
# fi

# 如果存在虚拟环境，可以取消下面几行的注释
# if [ -d "venv" ]; then
#     echo "激活venv虚拟环境..."
#     source venv/bin/activate
# fi

echo "运行MPE FOV环境测试..."
python test_mpe_fov.py

echo "测试完成!" 