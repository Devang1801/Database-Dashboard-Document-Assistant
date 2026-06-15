@echo off
REM Quick start script for Raksha Yantra Chatbot (Windows)

echo.
echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo  Raksha Yantra Chatbot - Quick Start (Windows)
echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version

REM Create virtual environment if not exists
if not exist "venv" (
    echo.
    echo  Creating virtual environment...
    python -m venv venv
)

echo ✓ Virtual environment ready

REM Install requirements
echo.
echo  Installing dependencies...
call venv\Scripts\activate.bat
pip install -q -r requirements.txt

if errorlevel 1 (
    echo  Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed

REM Check for environment variables
if not exist ".env" (
    echo.
    echo   No .env file found!
    echo Please create a .env file in this folder with:
    echo   DB_HOST=localhost
    echo   DB_USER=postgres
    echo   DB_PASSWORD=password
    echo   DB_NAME=ids
    echo   DB_PORT=5432
    echo   ANTHROPIC_API_KEY=your_key_here
    echo.
    pause
)

REM Start server
echo.
echo  Starting Chatbot Server...
echo Open your browser: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python server.py

pause
