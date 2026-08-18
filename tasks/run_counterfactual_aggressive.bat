@echo off
setlocal
cd /d "%~dp0\.."

echo [1/2] Checking training data (CPU mode)...
python tasks\run_counterfactual_training.py --stage check --device cpu
if errorlevel 1 goto :failed

echo [2/2] Starting head -^> last layer -^> last two layers (CPU mode)...
python tasks\run_counterfactual_training.py --stage aggressive --device cpu --no-amp --batch-size 2 --grad-accum 8 --eval-batch-size 16 --epochs 8 --workers 0
if errorlevel 1 goto :failed

echo.
echo Counterfactual training workflow completed.
exit /b 0

:failed
echo.
echo Workflow stopped. Read the FAIL or STOP message above.
exit /b 1
