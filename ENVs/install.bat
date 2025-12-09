@echo off
REM Installation script for ML-Agents 4.x Gymnasium Wrapper
REM For Windows

echo ================================================
echo Unity ML-Agents 4.x Gymnasium Wrapper Installer
echo ================================================
echo.

REM Check Python version
python --version
echo.

echo Checking if Python version is compatible (3.9-3.11)...
python -c "import sys; exit(0 if (3,9) <= sys.version_info[:2] <= (3,11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.9-3.11 is required for ML-Agents 4.x
    echo Current Python version is not compatible.
    pause
    exit /b 1
)

echo Python version is compatible!
echo.

echo Installing dependencies...
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch first (helps avoid conflicts)
echo Installing PyTorch...
pip install torch torchvision
echo.

REM Install requirements
echo Installing ML-Agents 4.x and other dependencies...
pip install -r requirements.txt
echo.

echo ================================================
echo Installation complete!
echo ================================================
echo.
echo Next steps:
echo 1. Open Unity Editor with your maze project
echo 2. Make sure your agent's Behavior Type is set to "Default"
echo 3. Run: python gymnasium_wrapper.py (to test connection)
echo 4. Run: python train_sb3.py --algorithm ppo --timesteps 500000
echo.
echo For more information, see README_GYMNASIUM.md
echo.

pause

