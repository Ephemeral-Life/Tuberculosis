@echo off
:: 进入虚拟环境的 Scripts 目录
cd /d "D:\python workspace\Tuberculosis\venv\Scripts"

:: 激活虚拟环境
call activate

:: 返回项目根目录
cd ../..

:: 执行 Python 脚本
python .\FTAD-TB\train_fd.py

:: 保持窗口打开
pause