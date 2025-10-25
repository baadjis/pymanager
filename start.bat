@echo off
REM start.bat - PyManager Startup Script for Windows
REM Starts both MCP server and Streamlit app

setlocal enabledelayedexpansion

REM Colors (limited in CMD but we'll use echo for clarity)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

cls
echo ========================================
echo    PyManager Startup
echo ========================================
echo.

REM Check Python installation
echo %INFO% Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %SUCCESS% Python %PYTHON_VERSION% found
echo.

REM Check required files
echo %INFO% Checking required files...
if not exist "mcp_server.py" (
    echo %ERROR% Missing mcp_server.py
    pause
    exit /b 1
)
if not exist "app.py" (
    echo %ERROR% Missing app.py
    pause
    exit /b 1
)
if not exist "requirements_mcp.txt" (
    echo %ERROR% Missing requirements_mcp.txt
    pause
    exit /b 1
)
echo %SUCCESS% All required files present
echo.

REM Check dependencies
echo %INFO% Checking Python dependencies...
python -c "import streamlit, fastapi, anthropic" 2>nul
if errorlevel 1 (
    echo %WARNING% Some dependencies missing. Installing...
    pip install -r requirements_mcp.txt
    if errorlevel 1 (
        echo %ERROR% Failed to install dependencies
        pause
        exit /b 1
    )
    echo %SUCCESS% Dependencies installed
) else (
    echo %SUCCESS% Dependencies OK
)
echo.

REM Check secrets file
echo %INFO% Checking configuration...
if not exist ".streamlit\secrets.toml" (
    echo %WARNING% No secrets.toml found
    echo %INFO% Creating default configuration...
    
    if not exist ".streamlit" mkdir ".streamlit"
    
    (
        echo # PyManager Configuration
        echo # Replace with your actual API key from https://console.anthropic.com/
        echo.
        echo ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
        echo MCP_SERVER_URL = "http://localhost:8000"
    ) > ".streamlit\secrets.toml"
    
    echo %SUCCESS% Created .streamlit\secrets.toml
    echo.
    echo %WARNING% IMPORTANT: Edit .streamlit\secrets.toml and add your Anthropic API key!
    echo.
    pause
)

REM Check if ports are available
echo %INFO% Checking port availability...

REM Check port 8000
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo %WARNING% Port 8000 is already in use
    set /p KILL_8000="Kill the process and continue? (Y/N): "
    if /i "!KILL_8000!"=="Y" (
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
            taskkill /F /PID %%a >nul 2>&1
        )
        echo %SUCCESS% Process on port 8000 killed
    ) else (
        echo %ERROR% Cannot start MCP server on port 8000
        pause
        exit /b 1
    )
)

REM Check port 8501
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo %WARNING% Port 8501 is already in use
    set /p KILL_8501="Kill the process and continue? (Y/N): "
    if /i "!KILL_8501!"=="Y" (
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
            taskkill /F /PID %%a >nul 2>&1
        )
        echo %SUCCESS% Process on port 8501 killed
    ) else (
        echo %ERROR% Cannot start Streamlit on port 8501
        pause
        exit /b 1
    )
)

echo %SUCCESS% Ports available
echo.

REM Start MCP Server
echo ========================================
echo    Starting MCP Server
echo ========================================
echo %INFO% Starting on http://localhost:8000
echo.

start /B python mcp_server.py > mcp_server.log 2>&1

REM Wait for MCP server to start
timeout /t 3 /nobreak >nul

REM Test MCP server
curl -s http://localhost:8000/ >nul 2>&1
if errorlevel 1 (
    echo %ERROR% MCP server not responding
    echo Check mcp_server.log for details
    type mcp_server.log
    pause
    exit /b 1
)

echo %SUCCESS% MCP server running
echo.

REM Start Streamlit App
echo ========================================
echo    Starting Streamlit App
echo ========================================
echo %INFO% Starting on http://localhost:8501
echo.

start /B streamlit run app.py > streamlit.log 2>&1

REM Wait for Streamlit to start
timeout /t 5 /nobreak >nul

echo %SUCCESS% Streamlit running
echo.

REM Print status
echo ========================================
echo    PyManager is Running!
echo ========================================
echo.
echo %SUCCESS% MCP Server: http://localhost:8000
echo %SUCCESS% Streamlit App: http://localhost:8501
echo.
echo %INFO% Logs:
echo   - MCP Server: mcp_server.log
echo   - Streamlit: streamlit.log
echo.
echo %INFO% Opening browser...
start http://localhost:8501
echo.

REM Run quick test
echo %INFO% Running connectivity test...
python test_mcp_integration.py --quick >nul 2>&1
if errorlevel 1 (
    echo %WARNING% Some tests failed, but services are running
    echo Run: python test_mcp_integration.py for details
) else (
    echo %SUCCESS% All systems operational!
)
echo.

echo ========================================
echo    Ready to Use!
echo ========================================
echo.
echo Navigate to AI Assistant page to start
echo.
echo Press any key to stop services and exit...
pause >nul

REM Cleanup
echo.
echo %INFO% Shutting down services...

REM Kill Python processes running our scripts
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /B "PID:"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo %SUCCESS% Services stopped
echo.
pause
exit /b 0
