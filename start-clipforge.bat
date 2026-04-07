@echo off
title ClipForge Pro Server
echo Starting ClipForge Pro...
echo.
cd /d D:\Apps\ClipForge_Pro\backend
call venv\Scripts\activate.bat
start https://clipforge-pro.vercel.app
echo Backend running at http://localhost:8000
echo API docs at http://localhost:8000/docs
echo Press Ctrl+C to stop.
echo.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
