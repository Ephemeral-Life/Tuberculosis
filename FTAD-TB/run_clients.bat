@echo off
setlocal enabledelayedexpansion

rem 设置 CUDA_VISIBLE_DEVICES 环境变量
set CUDA_VISIBLE_DEVICES=0

rem 循环启动客户端
for /L %%i in (0,1,4) do (
    echo Starting client.py with cid=%%i
    rem 在每个客户端启动前激活虚拟环境并切换目录
    start "Client %%i" cmd /c "call ..\venv\Scripts\activate && python client.py --cid %%i configs/symformer/symformer_retinanet_p2t_fpn_2x_TBX11K.py"
)

endlocal
