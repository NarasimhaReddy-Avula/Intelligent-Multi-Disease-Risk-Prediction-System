@echo off
REM Run the Streamlit webapp using the existing btp_env environment.
REM This avoids accidentally launching Streamlit from Anaconda/base Python.

setlocal

call C:\dev\venvs\btp_env\Scripts\activate.bat
cd /d "%~dp0"

REM Optional: load secrets from a local (non-OneDrive) file.
REM Create this file yourself if you want, e.g.:
REM   C:\dev\btp_secrets\btp_llm_env.bat
REM containing:
REM   set BTP_LLM_API_KEY=...
REM   set BTP_LLM_BASE_URL=https://api.groq.com/openai/v1
REM   set BTP_LLM_MODEL=llama-3.1-8b-instant
set "BTP_LLM_ENV_BAT=C:\dev\btp_secrets\btp_llm_env.bat"
if exist "%BTP_LLM_ENV_BAT%" call "%BTP_LLM_ENV_BAT%"

python -m streamlit run app.py
