@echo off
REM Streamlit UI 启动脚本 (CMD 批处理文件)
REM 用途：双击此文件即可启动 Streamlit 应用

REM 设置控制台编码为 UTF-8
chcp 65001 >nul

REM 获取脚本所在目录
cd /d "%~dp0"

REM 调用 PowerShell 脚本
powershell -ExecutionPolicy Bypass -File "%~dp0start_streamlit.ps1"

REM 如果 PowerShell 脚本执行失败，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo 启动失败，请查看上方错误信息
    pause
)
