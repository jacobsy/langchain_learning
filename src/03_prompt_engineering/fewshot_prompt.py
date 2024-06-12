from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

# 1. 创建一些示例 FewShot
samples = [

  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]


# 2. 创建一个提示模板
prompt_sample = PromptTemplate(input_variables=["flower_type", "occasion", "ad_copy"],
                               template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}")
#print(prompt_sample.format(**samples[0]))

# 3. 创建一个FewShotPromptTemplate对象

prompt = FewShotPromptTemplate(
    examples=samples,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"]
)
#print(prompt.format(flower_type="野玫瑰", occasion="爱情"))

# 4. 把提示传递给大模型
from dotenv import load_dotenv  # 用于加载环境变量
from langchain_community.chat_models import QianfanChatEndpoint

load_dotenv()  # 加载 .env 文件中的环境变量
model = QianfanChatEndpoint(streaming=True)
result = model.invoke(prompt.format(flower_type="野玫瑰", occasion="爱情"))
print(result)

# 5. 使用示例选择器
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

# 初始化示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    QianfanEmbeddingsEndpoint(),
    Chroma,
    k=1
)

# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"]
)
#print(prompt.format(flower_type="红玫瑰", occasion="爱情"))
result = model.invoke(prompt.format(flower_type="红玫瑰", occasion="爱情"))
print(result)