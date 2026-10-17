@echo off
setlocal enabledelayedexpansion
rem StrataPool: let the engine run at Medium integrity, so Windows large pages and the working-set lock work.
rem
rem A program starts at the LOWER of its parent's integrity and its file's mandatory label.  A folder shared with a
rem sandbox (Claude's Cowork, some sync tools) can carry a "Low" label that every new file in it inherits; an engine
rem started at Low gets "Lock pages in memory" stripped from its token, so large pages fail with error 1314 and Windows
rem pins less RAM.  This script:
rem   - relabels this folder's engine\*.exe, its .venv's Python and the Python that venv runs (pyvenv.cfg "home") to Medium,
rem     plus the same for a Strata install next to this folder (..\strata) and any folder or .exe you name;
rem   - removes the "downloaded from the internet" mark from the engine folder's files (copied from another PC);
rem   - says whether this account holds "Lock pages in memory".
rem
rem   POOL-INTEGRITY.bat                          this folder (and ..\strata if it exists)
rem   POOL-INTEGRITY.bat D:\ai\strata\.venv C:\x\y.exe   ... and these too (a venv folder, a folder or an .exe)
rem
rem Run it again after every rebuild or copy of the engine: a new strata.exe takes the folder's label again.
rem Right-click, "Run as administrator".  The folders themselves are left as they are.
net session >nul 2>&1
if errorlevel 1 (
  echo Right-click POOL-INTEGRITY.bat and choose "Run as administrator".
  pause
  exit /b 1
)
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set /a DONE=0, FAILED=0

echo Folder label:
call :showlabel "%ROOT%"
echo.

call :target "%ROOT%\engine"
call :target "%ROOT%\.venv"
if exist "%ROOT%\..\strata\" (
  for %%d in ("%ROOT%\..\strata") do set "SIB=%%~fd"
  call :target "!SIB!\engine"
  call :target "!SIB!\.venv"
)
:args
if "%~1"=="" goto checks
call :target "%~1"
shift
goto args

:checks
if exist "%ROOT%\engine\" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath '%ROOT%\engine' -File | Unblock-File" >nul 2>&1
  echo Removed the downloaded-file mark from engine\ ^(if it had one^).
)
echo.
whoami /priv | findstr /i "SeLockMemoryPrivilege" >nul
if errorlevel 1 (
  echo This account does NOT hold "Lock pages in memory", so large pages stay off.  To grant it:
  echo   secpol.msc, Local Policies, User Rights Assignment, "Lock pages in memory", add %USERNAME%,
  echo   then log off and on ^(the right reaches a new logon only^).
) else (
  echo This account holds "Lock pages in memory" ^(Disabled is normal: the engine turns it on^).
)
echo.
echo Relabelled %DONE% file^(s^), %FAILED% failed.  The engine's startup log should no longer say the large pages were
echo refused ^(error 1314^).
pause
exit /b 0

rem ---- a folder (engine or venv), or one .exe
:target
set "T=%~1"
if not exist "%T%" exit /b 0
if /i "%~x1"==".exe" (
  call :label "%T%"
  exit /b 0
)
for %%f in ("%T%\*.exe") do call :label "%%~ff"
if exist "%T%\Scripts\python.exe" (
  call :label "%T%\Scripts\python.exe"
  call :label "%T%\Scripts\pythonw.exe"
)
rem a venv's python.exe only launches the interpreter it was made from, which must not be Low either
if exist "%T%\pyvenv.cfg" (
  set "PYHOME="
  for /f "tokens=1,* delims==" %%a in ('findstr /b /i /c:"home" "%T%\pyvenv.cfg"') do (
    for /f "tokens=*" %%c in ("%%b") do set "PYHOME=%%c"
  )
  if defined PYHOME (
    rem a string test, not findstr: a piped echo would run without delayed expansion
    if "!PYHOME:WindowsApps=!"=="!PYHOME!" (
      call :label "!PYHOME!\python.exe"
      call :label "!PYHOME!\pythonw.exe"
    ) else (
      echo   skipped !PYHOME!: the Microsoft Store Python cannot be relabelled ^(use python.org's Python^)
    )
  )
)
exit /b 0

rem ---- one file to Medium
:label
if not exist "%~1" exit /b 0
icacls "%~1" /setintegritylevel M >nul 2>&1
if errorlevel 1 (
  echo   FAILED  %~1
  set /a FAILED+=1
  exit /b 0
)
set /a DONE+=1
echo   Medium  %~1
exit /b 0

rem ---- print a path's mandatory label (none shown = Medium, the default)
:showlabel
set "L="
for /f "delims=" %%l in ('icacls "%~1" 2^>nul ^| findstr /i "Mandatory"') do set "L=%%l"
if defined L (
  echo   %~1: !L:*Mandatory Label\=!
  if not "!L:Low Mandatory=!"=="!L!" echo   New files here inherit Low: run this again after each rebuild or copy.
) else (
  echo   %~1: no label ^(Medium^)
)
exit /b 0
