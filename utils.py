import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
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


def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool