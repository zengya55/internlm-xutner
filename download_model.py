import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from openxlab.model import download
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/home/xlab-app-center/model', revision='v1.0.3')
download(model_repo='zengya55/internlm-xtuner-zengya55', model_name='adapter_config.json', output='/home/xlab-app-center/model/hf/adapter_config.json')
download(model_repo='zengya55/internlm-xtuner-zengya55', model_name='README.md', output='/home/xlab-app-center/model/hf/README.md')
download(model_repo='zengya55/internlm-xtuner-zengya55', model_name='dapter_model.safetensors', output='/home/xlab-app-center/model/hf/dapter_model.safetensors')
download(model_repo='zengya55/internlm-xtuner-zengya55', model_name='xtuner_config.py', output='/home/xlab-app-center/model/hf/xtuner_config.py')

os.system('xtuner convert merge /home/xlab-app-center/model/Shanghai_AI_Laboratory/internlm-chat-7b /home/xlab-app-center/model/hf /home/xlab-app-center/model/merged --max-shard-size 2GB') 
os.system('xtuner chat /home/xlab-app-center/model/merged --bits 4 --prompt-template internlm_chat')