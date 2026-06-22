__import__('pysqlite3')
import sys
import streamlit as st
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
import warnings
import os
warnings.filterwarnings("ignore")

load_dotenv()

# https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/#chain-with-chat-history


llm = ChatOpenAI(api_key=st.secrets["openai_api_key"])
chat_history = []


# historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# to build the full QA chain
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

documents = TextLoader("./docs/faq.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings(api_key=st.secrets["openai_api_key"]))
retriever = vectorstore.as_retriever()

# Retrieve and generate using the relevant snippets of the blog.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


def generate_response(query):
    """ Generate a response to a user query"""
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
        )
    
    return rag_chain.invoke({
        "input": query, 
        "chat_history": chat_history})


def query(query):
    response = generate_response(query)
    # add this line to add to chat history
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    print("history", response["chat_history"])
    return response