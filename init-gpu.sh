#!/bin/bash

echo "====== INSTALLATION FOR GPU ENVIRONMENT IN PROGRESS ======"
conda install -c huggingface transformers
conda install pandas numpy scikit-learn matplotlib opencv plotly bokeh==2.4.1
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install annoy streamlit
echo "====== INSTALLATION FOR GPU ENVIRONMENT COMPLETED! ======"