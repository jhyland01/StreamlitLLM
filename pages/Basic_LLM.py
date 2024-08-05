import streamlit as st
import logging
import time
from utils import do_chat
from llama_index.llms.ollama import Ollama

if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")
    model = st.sidebar.selectbox("Choose a model", ["llama3", "phi3", "mistral"], index=0)
    llm = Ollama(model=model, request_timeout=300.0)
    logging.info(f"Model selected: {model}")

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
                        do_chat(llm, model, start_time)
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
