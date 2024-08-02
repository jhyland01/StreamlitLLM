import streamlit as st
from llama_index.llms.ollama import Ollama
import logging

def stream_chat(model, messages, doc_text=None):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        if doc_text:
            messages.append(ChatMessage(role="system", content=doc_text))
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e