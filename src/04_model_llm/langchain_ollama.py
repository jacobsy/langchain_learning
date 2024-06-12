# 导入 ollama 库
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen2:1.5b")
prompt = ChatPromptTemplate.from_template("请给我讲个{topic}的爱情故事")
res = llm.invoke(prompt.format(topic="曼陀罗"))
print(res)

