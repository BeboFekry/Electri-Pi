import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
# from pathlib import Path
import gdown

API = "AIzaSyBIvw7QEbrnN7HJTBqxu6CI_r7egCWf5tU"

class embedding:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, docs):
        embeddings = self.model.encode(docs)
        return embeddings.tolist()
    def embed_query(self, query):
        return self.model.encode(query).tolist()

file_id = '1-LspPqw7CET-euOzI8A3L60SQA3CQo9o'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'MyVectorDB/chroma.sqlite3'
gdown.download(url, output, quiet=False)

file_id = "1-XNpT7b_W6MR-9PyOyQT-4J4fnnzBdjO"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'MyVectorDB/785d685c-8ac5-4be7-9e32-13b827c88a50/data_level0.bin'
gdown.download(url, output, quiet=False)

file_id = "1-cYtH2qkE-3ouIAZuTKsprLo00i0WEiI"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'MyVectorDB/785d685c-8ac5-4be7-9e32-13b827c88a50/header.bin'
gdown.download(url, output, quiet=False)

file_id = "1-QpaRSHPlF-6GYivut1HGe9L0Ec-9pAi"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'MyVectorDB/785d685c-8ac5-4be7-9e32-13b827c88a50/index_metadata.pickle'
gdown.download(url, output, quiet=False)

file_id = "1-jwYjTVXbJT75Kvme70EWosCfkfV8p__"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'MyVectorDB/785d685c-8ac5-4be7-9e32-13b827c88a50/length.bin'
gdown.download(url, output, quiet=False)

file_id = "1-jYB22FMgoYND6D34Rp2mHDsT_C6YL5N"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'MyVectorDB/785d685c-8ac5-4be7-9e32-13b827c88a50/link_lists.bin'
gdown.download(url, output, quiet=False)

embed_model = embedding()

vector_database = Chroma(persist_directory="/MyVectorDB", embedding_function=embed_model)
retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={'k':3})

llm = GoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=API,temprature=0)

temp = """you are an AI helpfull assistant that helps users by answering their questions in Arabic
  you need to add two outputs by writing the retrieved document and then answering the question
  knowlege you know:
  {context}
  Question: {question}"""
prompt = PromptTemplate.from_template(temp)
rag_chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.columns([1,1,1])[1].image("images/icon.png")
st.header("Caroline")
st.info("I Care chatbot for medical diagnosis - Easy Healthcare for Anyone Anytime")


def predict(m):
    output = rag_chain.invoke(m)
    st.chat_message("user").markdown(m)
    st.session_state.messages.append({"role": "user", "content": m})
    st.chat_message("assistant").markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})

message = st.chat_input("Type Your Prompt.")

if message is None:
    pass
else:
    predict(message)
