@echo off
setlocal enabledelayedexpansion
rem StrataPool: hold this PC's NVIDIA GPU(s) at full clocks while the pool runs (Windows).
rem In a split each PC's GPU waits while the other PCs run their layers, and the Windows driver lowers its clocks in
rem those gaps: a worker RTX 3070 spent 94% of a decode run below full clocks, and its share of a window took 12-22 ms
rem instead of 3.7-4.1 ms locked (lask3802's measurement, Strata PR #1748).  Linux drivers keep full clocks.
rem
rem   POOL-CLOCKS.bat          lock: graphics between 75% of its maximum and the maximum, memory at its maximum
rem   POOL-CLOCKS.bat reset    give the clocks back to the driver (a reboot does it too)
rem
rem Right-click, "Run as administrator".  A card or driver that refuses a lock says so and is left as it was.
net session >nul 2>&1
if errorlevel 1 (
  echo Right-click POOL-CLOCKS.bat and choose "Run as administrator".
  pause
  exit /b 1
)
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo nvidia-smi was not found: is the NVIDIA driver installed?
  pause
  exit /b 1
)
if /i "%~1"=="reset" (
  nvidia-smi -rgc
  nvidia-smi -rmc
  echo.
  echo The clocks are the driver's again.
  pause
  exit /b 0
)
for /f "tokens=1-3 delims=, " %%a in ('nvidia-smi --query-gpu^=index^,clocks.max.graphics^,clocks.max.memory --format^=csv^,noheader^,nounits') do (
  set /a GMIN=%%b*3/4
  echo GPU %%a: graphics !GMIN!-%%b MHz, memory %%c MHz
  nvidia-smi -i %%a -lgc !GMIN!,%%b
  nvidia-smi -i %%a -lmc %%c,%%c
)
echo.
echo Locked until "POOL-CLOCKS.bat reset" or a reboot.  Check while a pool request runs:
echo   nvidia-smi --query-gpu=pstate,clocks.gr,clocks.mem --format=csv -l 1
pause
