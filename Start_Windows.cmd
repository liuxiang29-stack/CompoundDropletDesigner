@echo off
cd /d "%~dp0"
python app\compound_droplet_designer.py
if errorlevel 1 pause
