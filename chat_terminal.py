#!/usr/bin/env python
from typing import List
from dotenv import load_dotenv
import sys

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
import argparse

def get_model(model_name: str):
  print(f"Running with {model_name} ====")
  if model_name == "ollama":
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    llm = Ollama(model="llama2")
    embeddings = OllamaEmbeddings()
    return llm, embeddings
  if model_name == "openai":
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings()
    return llm, embeddings

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Running LangChain')
    parser.add_argument('input', type=str, help='Input to the model')
    parser.add_argument('-l', '--local', action="store_true", help='Run with local model')
    args = parser.parse_args()

    model_name = "ollama" if args.local else "openai"
    llm, embeddings = get_model(model_name)
    
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    loader = DirectoryLoader("data/") # not working
    documents = TextLoader("data/dog.txt").load()
    if not documents:
        print("No documents found in data directory")
        sys.exit()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embeddings) 
    retriever = db.as_retriever(search_kwargs={"k": 1})

    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    # Prompt to answer the question
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])  
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    chat_history = []
    print("initial query", args.input)
    query = args.input
    while True:
      if not query:
        query = input("Prompt: ")
      if query in ['quit', 'q', 'exit']:
        sys.exit()
      result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
      print("result", result['answer'])

      chat_history.append((query, result['answer']))
      query = None

if __name__ == "__main__":
    main()


