@echo off
title Road Safety Assistant
cd /d "%~dp0"

echo ======================================================================
echo Starting Vision-Based Traffic Sign Recognition App...
echo ======================================================================

:: Start Python Flask Backend in background
start /B py -3.11 app.py > nul 2>&1

:: Wait 1.5 seconds for server to initialize
timeout /t 2 /nobreak > nul

:: Launch in pure standalone Desktop App Window (No address bar, No browser tabs)
start msedge --app="http://127.0.0.1:5000" --window-size=1300,880 || start chrome --app="http://127.0.0.1:5000" --window-size=1300,880 || py -3.11 desktop_app.py

exit
