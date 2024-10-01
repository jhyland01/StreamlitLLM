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



nest_asyncio.apply()

upload_dir = "./data"
os.makedirs(upload_dir, exist_ok=True)

st.title("Query documents with a local RAG model")

if uploaded_files := st.file_uploader("Upload your files", accept_multiple_files=True):
    for uploaded_file in uploaded_files:
        with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(
        f"Files uploaded successfully. Files in folder: {os.listdir(upload_dir)}"
    )

paper_to_tools_dict = {}
for paper in uploaded_files:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in uploaded_files for t in paper_to_tools_dict[paper]]

model_name = st.sidebar.selectbox(
    "Choose a model", ["llama3", "phi3", "mistral"], index=0
)

llm = Ollama(model=model_name, request_timeout=300.0)

embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
    embed_model=embedding_model,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True,
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
