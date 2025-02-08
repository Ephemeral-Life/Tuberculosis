#!/bin/bash

# 启动30个客户端，每个在后台运行并记录日志
for cid in {0..29}; do
    CUDA_VISIBLE_DEVICES=0 python client.py --cid $cid configs/symformer/symformer_retinanet_p2t_fpn_2x_TBX11K.py > "client_${cid}.log" 2>&1 &
done

# 等待所有后台任务完成
wait
echo "所有客户端任务已完成"