@echo off
cd /d "%~dp0"
call "venv\Scripts\activate"
set port=8082
:loop
ray start --head --port=%port%
if %errorlevel% == 0 goto continue
ray status >nul 2>&1
if %errorlevel% == 0 goto continue
set /a port=%port% + 1
goto loop
:continue
python "FTAD-TB\train_fd.py"
cmd /k