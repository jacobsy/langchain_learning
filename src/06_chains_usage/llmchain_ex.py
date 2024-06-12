# 导入所需的库
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import QianfanLLMEndpoint
from langchain.chains import LLMChain
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
# 原始字符串模板
template = "{flower}的花语是?"
# 创建模型实例
llm = QianfanLLMEndpoint(streaming=True)

# 创建LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template))

# 调用LLMChain，返回结果
result = llm_chain("玫瑰")
print(result)