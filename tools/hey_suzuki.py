import tempfile
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import SequentialChain

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.callbacks.base import BaseCallbackHandler


import os
#Enter open AI Key 
os.environ['OPENAI_API_KEY'] = ''



class suzuki_RAG:

    def __init__(self):
        
        self.DB_FAISS_PATH = 'vectorstore/'

    def conversational_chat(self,query,top_k,tempr):
        
        DB_FAISS_PATH = 'vectorstore/'

        
        embeddings = OpenAIEmbeddings()
        
        db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization = True)
        similarity_data = db.similarity_search(query)

        db_s_f = FAISS.from_texts([similarity_data[0].page_content] , embeddings)
        
        for i in range(top_k-1):
            db_s_i = FAISS.from_texts([similarity_data[i+1].page_content] , embeddings)
            db_s_f.merge_from(db_s_i)
        
        retriever = db_s_f.as_retriever()

        db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization = True)

        retriever = db.as_retriever()

        template = """
                    You are a digital automobile assistant of Maruti Suzuki India Limited 
                    Use the following context to answer the users's question 
                    {context}

                    Users question: {question}

                    You are a digital assistant keep your answer conversational.
                    """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        model = ChatOpenAI(temperature = tempr)
       

        chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                    )
        answer = chain.invoke(query)
        
        return answer