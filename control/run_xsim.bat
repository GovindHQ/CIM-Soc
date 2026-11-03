@echo off
rem ---------------------------------------------------------------------------
rem run_xsim.bat -- simulate the control path with Vivado XSim.
rem
rem   run_xsim.bat              NUM_ARRAYS=1, NUM_TOKENS=4
rem   run_xsim.bat 2            NUM_ARRAYS=2  (testbench then uses array 1)
rem   run_xsim.bat 1 1          one token: the original single-MVM demo
rem
rem Needs Vivado's bin directory on PATH (xvlog / xelab / xsim).
rem ---------------------------------------------------------------------------
setlocal
set NARR=%1
if "%NARR%"=="" set NARR=1
set NTOK=%2
if "%NTOK%"=="" set NTOK=4
set TAG=a%NARR%_t%NTOK%

cd /d "%~dp0"
if not exist build mkdir build
cd build

call xvlog ..\token_buffer.v ..\weight_memory.v ..\result_buffer.v ..\cim_array.v ^
           ..\array_controller.v ..\cim_top.v ..\tb_control.v || exit /b 1
call xelab tb_control -generic_top "NUM_ARRAYS=%NARR%" -generic_top "NUM_TOKENS=%NTOK%" -s sim_%TAG% || exit /b 1
call xsim sim_%TAG% -R -log sim_%TAG%.log || exit /b 1
findstr /C:"RESULT: PASS" sim_%TAG%.log >nul || exit /b 1
