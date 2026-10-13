@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo Strata Custom Engine Builder (Dual RTX 3060 sm_86)
echo Building from /Source and deploying to /App/engine
echo ========================================================

set "VCVARS="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if "%VCVARS%"=="" (
    echo [ERROR] MSVC vcvars64.bat not found. Please install MSVC Build Tools.
    exit /b 1
)

echo Loading MSVC 64-bit environment from: %VCVARS%
call "%VCVARS%"

set "NINJA_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if exist "%NINJA_DIR%" set "PATH=%NINJA_DIR%;%PATH%"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=%CUDA_PATH%\bin;%PATH%"

cd /d "%~dp0"

echo Cleaning previous build directory...
if exist build rd /s /q build
mkdir build
cd build

echo Configuring CMake with Ninja for sm_86...
cmake .. ^
  -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DSTRATA_NATIVE_EXPERTS=ON ^
  -DSTRATA_BUILD_TESTS=OFF

if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo Compiling strata.exe with Ninja...
ninja strata

if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo Deploying new strata.exe to ..\..\App\engine...
if not exist "..\..\App\engine" mkdir "..\..\App\engine"
copy /y "strata.exe" "..\..\App\engine\strata.exe"

echo [SUCCESS] Custom sm_86 Strata engine deployed to App\engine\strata.exe!
