import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr
os.system('python download_model.py')
os.system('xtuner convert merge /home/xlab-app-center/model/Shanghai_AI_Laboratory/internlm-chat-7b /home/xlab-app-center/model/hf /home/xlab-app-center/model/merged --max-shard-size 2GB') 
model_name_or_path = "/home/xlab-app-center/model/merged"
# 检查路径是否存在
if os.path.exists(model_name_or_path):
    # 如果路径存在，进一步检查它是否是文件夹
    if os.path.isdir(model_name_or_path):
        print(f"'{model_name_or_path}' exists and it's a directory.")
    else:
        print(f"'{model_name_or_path}' exists but it's not a directory, it might be a file.")
else:
    print(f"'{model_name_or_path}' does not exist.")


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model finetuned that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
# messages = [(system_prompt, '')]
# input_text = input("User  >>> ")
# input_text = input_text.replace(' ', '')
# response, history = model.chat(tokenizer, input_text, history=messages)
# 

class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        #self.chain = load_chain()
        pass

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            response, history = model.chat(tokenizer, question, history=chat_history)
            chat_history.append(
                (question, response))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history


# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch()
