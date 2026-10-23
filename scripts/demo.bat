@echo off
rem Play a video (file, RTSP/HTTP URL or webcam index) through Stage A + Stage B and show the result.
rem   scripts\demo.bat video.mp4 [options of scripts\demo_video.py, e.g. --save result.mp4]
rem Uses the active Python environment (or set PYTHON=path\to\python.exe) and the checkpoint
rem runs\main\seed42\model.pt (or set STAGE_B=path\to\model.pt).
if "%~1"=="" (
  echo usage: scripts\demo.bat ^<video ^| rtsp url ^| webcam index^> [options]
  exit /b 1
)
setlocal
if not defined PYTHON set PYTHON=python
if not defined STAGE_B set STAGE_B=%~dp0..\runs\main\seed42\model.pt
if not exist "%STAGE_B%" (
  echo Stage B checkpoint not found: %STAGE_B%
  exit /b 1
)
"%PYTHON%" "%~dp0demo_video.py" --source "%~1" --stage-b "%STAGE_B%" %2 %3 %4 %5 %6 %7 %8 %9
