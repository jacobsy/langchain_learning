from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

class ChatbotWithRetrieval:
    def __init__(self, dir):
        base_dir = dir
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        # 向量数据库
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits,  # 以分块的文档
            embedding=QianfanEmbeddingsEndpoint(),  # 用Baidu Qianfan的Embedding Model做嵌入
            location=":memory:",  # in-memory 存储
            collection_name="my_documents", )  # 指定collection_name

        # 初始化LLM
        self.llm = QianfanChatEndpoint(streaming=True)

        # 初始化Memory
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True
        )

        # 初始化对话历史
        self.conversation_history = ""

        # 设置Retrieval Chain
        retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=retriever,
            memory=self.memory
        )

    def get_response(self, user_input):
        response = self.qa(user_input)
        # 更新对话历史
        self.conversation_history += f"你: {user_input}\nChatbot: {response['answer']}\n"
        return self.conversation_history

if __name__ == "__main__":
    folder = "D:\\pyworkspace\\langchain_learning\\docs"
    bot = ChatbotWithRetrieval(folder)

    # 定义 Gradio 界面
    interface = gr.Interface(
        fn=bot.get_response,  # 使用我们刚刚创建的函数
        inputs="text",  # 输入是文本
        outputs="text",  # 输出也是文本
        live=False,  # 实时更新，这样用户可以连续与模型交互
        title="花舞阁鲜花智能客服",  # 界面标题
        description="请输入问题，然后点击提交。"  # 描述
    )
    interface.launch()  # 启动 Gradio 界面