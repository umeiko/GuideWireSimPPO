#!/bin/bash
sudo apt update
sudo apt install p7zip-full -y
find ./datas/ -type f ! -name "*zip*" -delete
find ./datas/* -type d ! -name "*zip*" -delete

cd ./datas

save_dir="./"

# 定义下载链接
links=(
    "https://gitee.com/meiziyang2023/g_env/releases/download/dataset_1.0/datas.zip.003" 
    "https://gitee.com/meiziyang2023/g_env/releases/download/dataset_1.0/datas.zip.002" 
    "https://gitee.com/meiziyang2023/g_env/releases/download/dataset_1.0/datas.zip.001"
    )

# 遍历链接列表
for link in "${links[@]}"; do
  # 获取文件名
  filename=$(basename "$link")

  # 检查文件是否已下载
  if [ ! -f "$save_dir/$filename" ]; then
    echo "Downloading $filename..."
    # 下载文件
    wget "$link" -O "$save_dir/$filename"

    # 检查文件是否下载成功
    if [ $? -ne 0 ]; then
      echo "Failed to download $filename."
      exit 1
    fi
  fi
done

7z x datas.zip.001

rm -rf datas.zip.*
mv ./datas/* ./
rm -r ./datas

# cd ..
# cd ./env
# git clone https://gitee.com/meiziyang2023/g_env.git
