@echo off
REM powershell -Command "Start-Process 'streamlit' -ArgumentList 'run MyChatbot.py --server.headless true' -WindowStyle Hidden"
streamlit run MyChatbot.py --server.headless true