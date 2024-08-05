import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
import logging
import time
import os
from utils import do_rag_chat

logging.basicConfig(level=logging.INFO)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Directory for uploaded files
upload_dir = "./data"
os.makedirs(upload_dir, exist_ok=True)

def load_documents():
    return SimpleDirectoryReader(upload_dir).load_data()

if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")
    model = st.sidebar.selectbox("Choose a model", ["llama3", "phi3", "mistral"], index=0)
    logging.info(f"Model selected: {model}")
    llm = Ollama(model=model, request_timeout=300.0)
    Settings.llm = llm

    uploaded_file = st.file_uploader("Drag and drop a file here", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

    if documents := load_documents():
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm=llm)
    else:
        st.error("No documents found in the directory.")
        logging.error("No documents found in the directory.")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        do_rag_chat(llm, query_engine, prompt, model, start_time)
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
