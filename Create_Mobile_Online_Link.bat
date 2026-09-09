@echo off
title Traffic Sign AI - Mobile Online Tunnel
cd /d "%~dp0"

echo ======================================================================
echo Starting Road Safety Assistant Server & Secure Mobile Tunnel (HTTPS)...
echo ======================================================================

:: Start Flask server in background
start /B py -3.11 app.py > nul 2>&1

timeout /t 2 /nobreak > nul

echo.
echo ======================================================================
echo YOUR SECURE MOBILE URL (HTTPS):
echo (Use this HTTPS URL in WebIntoApp or open directly on any phone!)
echo ======================================================================
cmd /c npx localtunnel --port 5000
pause
