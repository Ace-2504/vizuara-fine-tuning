@echo off
REM ============================================================================
REM  Resume the local Gemma evaluation on the RTX 3060.
REM  Safe to run again after a power cut / crash / Ctrl-C: it skips every item and
REM  every version already finished, and continues exactly where it stopped.
REM  Double-click this file, or run it from a terminal.
REM ============================================================================
cd /d "%~dp0"
".venv-cuda\Scripts\python.exe" "evaluations\eval_local.py" %*
echo.
echo ---------------------------------------------------------------------------
echo  If it stopped early (power / crash), just run this file again to resume.
echo  When all 4 gemma results are in eval_results\, the judge + report come next.
echo ---------------------------------------------------------------------------
pause
