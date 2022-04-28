#!/bin/bash
# starts the app and restarts it if crashed

while true; do
    [ -e stopme ] && break
    streamlit run ./app.py --server.enableCORS False
done