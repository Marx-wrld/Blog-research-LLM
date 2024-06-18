import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv() # take env variables from .env

st.title('News Research Tool 📊')

st.sidebar.title('News Article URLs 🌎')

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1} 📰")
    urls.append(url)
    
process_url_clicked = st.sidebar.button("Process URLs 🔎")
file_path = 'faiss_store_openai.pkl'

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started...🔄")
    data = loader.load() #loading data
    #Splitting the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', '.'],
        chunk_size=1000
    )
    main_placeholder.text("Text splitter started...🔄")
    docs = text_splitter.split_documents(data) #our chunks will go here
    # create embeddings and save to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding vector started...🔄")
    time.sleep(2)
    
    # save the FAISS index to a pickle file
    with open(file_path, 'pb') as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retrieve=vectorstore.as_retriever())
            result = chain({"question":query}, return_only_outputs=True)
            # result = {"answer": "", "sources": []}
            st.header("Answer")
            st.subheader(result["answer"])
            
            #Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") #Split the sources by newline
                for source in sources_list:
                    st.write(source)