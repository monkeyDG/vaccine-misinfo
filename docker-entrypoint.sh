#!/bin/bash
cd /home
nohup flask run --host=0.0.0.0 > app.log &
nohup streamlit run streamlit_ui.py --server.port 80 > streamlit_ui.log