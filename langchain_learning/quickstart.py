'''
 调用百度千帆大模型，生成简单文本
'''

import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

chat = QianfanChatEndpoint(
    streaming=True
)
res = chat([HumanMessage(content="请给我的书店起一个名字")])

print(res.content)