#!/bin/bash

echo "====== INSTALLATION FOR CPU ENVIRONMENT IN PROGRESS ======"
conda install -c huggingface transformers
conda install pandas numpy scikit-learn matplotlib opencv plotly bokeh==2.4.1
pip install torch torchvision annoy streamlit
echo "====== INSTALLATION FOR CPU ENVIRONMENT COMPLETED! ======"