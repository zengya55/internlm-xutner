import os
os.system('export TOKENIZERS_PARALLELISM=false') 
os.system('streamlit run app.py --server.address=0.0.0.0 --server.port 7860')
