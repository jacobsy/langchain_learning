# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate
# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template)
# 打印LangChain提示模板的内容
print(prompt)

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_community.chat_models import QianfanChatEndpoint

chat = QianfanChatEndpoint(
    streaming=True
)

# 输入提示
input = prompt.format(flower_name=["康乃馨"], price='50')
# 得到模型的输出
output = chat.invoke(input)
print(output)