@echo off
setlocal

cd /d %~dp0
set MODE=%1
if "%MODE%"=="" set MODE=core

echo ============================================================
echo FAPR-Depth v6 paper analysis suite
echo Mode: %MODE%
echo ============================================================

if /I "%MODE%"=="core" goto CORE
if /I "%MODE%"=="stress-quick" goto STRESS_QUICK
if /I "%MODE%"=="stress-paper" goto STRESS_PAPER
if /I "%MODE%"=="all" goto ALL

echo Unknown mode: %MODE%
echo Usage:
echo   run_fapr_v6_analysis_suite.bat core
echo   run_fapr_v6_analysis_suite.bat stress-quick
echo   run_fapr_v6_analysis_suite.bat stress-paper
echo   run_fapr_v6_analysis_suite.bat all
exit /b 2

:CORE
call :RUN_CORE
exit /b %ERRORLEVEL%

:STRESS_QUICK
python stress_test_fapr_relative_prior.py --profile quick --max-shards 64
exit /b %ERRORLEVEL%

:STRESS_PAPER
python stress_test_fapr_relative_prior.py --profile paper --max-shards 512
exit /b %ERRORLEVEL%

:ALL
call :RUN_CORE
if errorlevel 1 exit /b %ERRORLEVEL%
python stress_test_fapr_relative_prior.py --profile paper --max-shards 512
exit /b %ERRORLEVEL%

:RUN_CORE
python analyze_fapr_failure_posterior.py
if errorlevel 1 exit /b %ERRORLEVEL%
python analyze_fapr_failure_regions.py
if errorlevel 1 exit /b %ERRORLEVEL%
python visualize_fapr_expert_routing.py
if errorlevel 1 exit /b %ERRORLEVEL%
python analyze_fapr_safe_correction.py --phase joint
if errorlevel 1 exit /b %ERRORLEVEL%
python analyze_fapr_risk_calibration.py --phase joint
if errorlevel 1 exit /b %ERRORLEVEL%
exit /b 0
