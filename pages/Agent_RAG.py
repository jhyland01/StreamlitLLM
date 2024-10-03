import nest_asyncio
import os
from utils import get_doc_tools
from pathlib import Path
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage


nest_asyncio.apply()

upload_dir = "./data"
os.makedirs(upload_dir, exist_ok=True)

st.title("Query documents with a local acentic RAG model")

if uploaded_files := st.file_uploader("Upload your files", accept_multiple_files=True):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded successfully: {uploaded_file.name}")

# get relative path of tiles in upload_dir
file_list = [
    os.path.join(upload_dir, file_name) for file_name in os.listdir(upload_dir)
]

model_name = st.sidebar.selectbox(
    "Choose a model", ["llama3.1", "phi3", "mistral"], index=0
)

llm = Ollama(model=model_name, request_timeout=300.0)

embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

paper_to_tools_dict = {}
for paper in file_list:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem, llm)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in file_list for t in paper_to_tools_dict[paper]]

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
    embed_model=embedding_model,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt=""" \
You are an agent designed to answer queries over a set of given documents.
Please always use the tools provided to answer a question. Do not rely on prior knowledge. Make sure your final response is appropriate for an end business user. Your response should not mention the tools used but provide a useable summary of what you have found.\

""",
    verbose=True,
)
agent = AgentRunner(agent_worker)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]


user_input = st.text_input("You: ", key="user_input")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    conversation_history = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.messages
        ]
    )

    agent_response = agent.query(conversation_history)

    # Log and display the agent's response
    st.session_state.messages.append(
        {"role": "assistant", "content": agent_response.response}
    )

# Display the chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"You: {msg['content']}")
    else:
        st.write(f"Assistant: {msg['content']}")
