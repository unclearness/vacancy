rd /s /q win_build
md win_build
cd win_build

IF EXIST "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\" (
cmake -G "Visual Studio 16 2019" ..
) ELSE IF EXIST "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\" (
cmake -G "Visual Studio 15 2017 Win64" ..
)

pause
