import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time

def stream_chat(llm, model, messages, doc_text=None):
    try:
        if doc_text:
            messages.append(ChatMessage(role="system", content=doc_text))
        response = ""
        response_placeholder = st.empty()
        for r in llm.stream_chat(messages):
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def append_response_and_log(response_message, start_time):
    duration = time.time() - start_time
    response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
    st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
    st.write(f"Duration: {duration:.2f} seconds")
    logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

def do_rag_chat(llm, query_engine, prompt, model, start_time):
    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
    retrieved_docs = query_engine.query(prompt)
    retrieved_docs_text = "\n".join(doc.text for doc in retrieved_docs)
    print(retrieved_docs_text)
    response_message = stream_chat(llm, model, messages, retrieved_docs_text)
    append_response_and_log(response_message, start_time)

def do_chat(llm, model, start_time):
    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
    response_message = stream_chat(llm, model, messages)
    append_response_and_log(response_message, start_time)
