import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from openxlab.model import download
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/home/xlab-app-center/model', revision='v1.0.3')
download(model_repo='zengya55/internlm-xtuner-zengya55', output='/home/xlab-app-center/model/hf')

