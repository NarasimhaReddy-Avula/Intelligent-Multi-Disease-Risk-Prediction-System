@echo off
:: BTP Project Environment Activation Script
:: Location: C:\dev\venvs\btp_env (outside OneDrive to prevent sync issues)

echo.
echo ========================================
echo   BTP Project Environment Activation
echo ========================================
echo.

:: Activate the virtual environment
call C:\dev\venvs\btp_env\Scripts\activate.bat

:: Navigate to project folder
cd /d "C:\Users\DELL\OneDrive\ドキュメント\BTP\PROJECT"

echo Environment activated!
echo Location: C:\dev\venvs\btp_env
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.
echo Quick commands:
echo   python --version    - Check Python version
echo   jupyter lab         - Start Jupyter Lab
echo   streamlit run app   - Run Streamlit app
echo   deactivate          - Exit virtual environment
echo.
