# 🎯 Streamlit Cloud లో ChromaDB క్రాష్ అవ్వకుండా ఉండే బైపాస్ (దీన్ని అందరికంటే పైనే ఉంచాలి)
__import__('pysqlite3')
import sys
import streamlit as st
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import warnings
import logging
from dotenv import load_dotenv

# వార్నింగ్స్ ని బ్లాక్ చేయడం
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("langchain_text_splitters.character").setLevel(logging.ERROR)

load_dotenv()

# 🎯 LangChain v0.3+ పక్కా ఇంపోర్ట్స్ (Chroma ఇక్కడే ఉండాలి, chains ఎర్రర్స్ రాకుండా బైపాస్)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 1. Models & Streamlit Secrets Setup
# st.secrets నుండి సురక్షితంగా API Key ని తీసుకుంటున్నాం
api_key = st.secrets["openai_api_key"]

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1", 
    model="openrouter/auto", 
    temperature=0,
    api_key=api_key
)
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=api_key
)

# historical messages and the latest user question
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

# build the full QA chain
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- 2. Indexing (Data Setup - 100% Streamlit Cloud Safe) ---
documents = TextLoader("./docs/faq.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n")
splits = text_splitter.split_documents(documents)

# 🎯 పాత కోడ్ లో ఉన్న లోపాన్ని సరిచేసి ఇక్కడ 'splits' ని పంపాము
db = Chroma.from_documents(splits, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 1})

# --- 3. Pure LCEL Pipeline (പాత chains స్థానంలో మోడరన్ పైప్‌లైన్) ---
contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(
            contextualize_chain.invoke({"input": x["input"], "chat_history": x["chat_history"]})
            if x["chat_history"] else x["input"]
        )
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)


def generate_response(query_text, history_list):
    """ Streamlit సెషన్ హిస్టరీని బట్టి సమాధానం తెస్తుంది """
    ai_response = rag_chain.invoke({"input": query_text, "chat_history": history_list})
    return ai_response


def query(user_query):
    """ 
    Streamlit యాప్ ఈ ఫంక్షన్ ని పిలుస్తుంది. 
    ఇది ప్రతి యూజర్ కి విడివిడిగా 'st.session_state' ద్వారా ల్యాంగ్‌చైన్ మెమరీని హ్యాండిల్ చేస్తుంది.
    """
    # 🎯 ప్రతి యూజర్ కి విడివిడిగా ల్యాంగ్‌చైన్ ఫార్మాట్ లో హిస్టరీ ని మెయింటైన్ చేయడానికి:
    if "langchain_history" not in st.session_state:
        st.session_state.langchain_history = []
        
    # రెస్పాన్స్ జనరేట్ చేయడం
    response_text = generate_response(user_query, st.session_state.langchain_history)
    
    # 🎯 ల్యాంగ్‌చైన్ మెమరీ ని అప్‌డేట్ చేయడం
    st.session_state.langchain_history.append(HumanMessage(content=user_query))
    st.session_state.langchain_history.append(AIMessage(content=response_text))
    
    # app.py లోని response["answer"] స్ట్రక్చర్ కి మ్యాచ్ అయ్యేలా రిటర్న్ చేస్తున్నాం
    return {"answer": response_text}