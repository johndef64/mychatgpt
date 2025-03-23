@echo off
powershell -Command "Start-Process 'streamlit' -ArgumentList 'run MyChatbot.py --server.headless true' -WindowStyle Hidden"
REM streamlit run MyChatbot.py --server.headless true