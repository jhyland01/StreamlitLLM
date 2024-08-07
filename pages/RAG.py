import os
import asyncio
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

model_name = st.sidebar.selectbox("Choose a model", ["llama3", "phi3", "mistral"], index=0)

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""
    nodes: list[NodeWithScore]

upload_dir = "./data"
os.makedirs(upload_dir, exist_ok=True)

class RAGWorkflow(Workflow):
    def __init__(self, model_name): 
        self.model_name = model_name

    @step(pass_context=True)
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = './data'
        if not dirname:
            return None

        documents = SimpleDirectoryReader(dirname).load_data()
        ctx.data["index"] = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        )
        return StopEvent(result=f"Indexed {len(documents)} documents.")

    @step(pass_context=True)
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        ctx.data["query"] = query

        # get the index from the global context
        index = ctx.data.get("index")
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step(pass_context=True)
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = Ollama(model=self.model_name, request_timeout=300.0)
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = ctx.data.get("query")

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        result_str = await response._async_str()  # Ensure this is awaited
        return StopEvent(result=result_str)
    
workflow = RAGWorkflow(model_name=model_name)

async def run_workflow(query):
    ctx = Context()
    await workflow.ingest(ctx, StartEvent())
    retriever_event = await workflow.retrieve(ctx, StartEvent(query=query))
    stop_event = await workflow.synthesize(ctx, retriever_event)
    return stop_event.result

st.title("Query documents with a local RAG model")

if uploaded_files := st.file_uploader("Upload your files", accept_multiple_files=True):
    for uploaded_file in uploaded_files:
        with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully.")

query = st.text_input("Enter your query:")

if st.button("Run Query"):
    if query:
        result = asyncio.run(run_workflow(query))
        st.write("Query Result:")
        st.write(result)
    else:
        st.error("Please enter a query.")
