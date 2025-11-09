@echo off
setlocal

python data\make_data.py
if errorlevel 1 exit /b %errorlevel%

python models\train.py
if errorlevel 1 exit /b %errorlevel%

python models\tune.py
if errorlevel 1 exit /b %errorlevel%

python models\evaluate.py
if errorlevel 1 exit /b %errorlevel%

python app\app.py

