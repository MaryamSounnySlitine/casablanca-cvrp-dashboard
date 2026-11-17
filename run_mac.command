#!/bin/sh
cd "$(dirname "$0")"
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run app.py
