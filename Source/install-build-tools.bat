@echo off
echo ========================================================
echo Installing Visual Studio 2022 C++ Build Tools
echo ========================================================
echo NOTE: Must be run with Administrator privileges.
echo.
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
echo.
echo Installation finished. You can now run build-custom-engine.bat.
pause
