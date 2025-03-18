
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import base64
import os
from langchain import hub
from dotenv import load_dotenv
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph


dotenv_path = Path('/home/tamil/work/Dohts/src/dohts/creds.env')
load_dotenv(dotenv_path=dotenv_path)

# Your credentials
username = os.getenv('user')
password = os.getenv('password')
print(username)
print(password)

# Encode credentials in Base64
credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

# Define headers with Basic Authentication
headers = {
            "Authorization": f"Basic {credentials}"
            }
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv('base_url'), 
                                client_kwargs={'headers': headers})

new_vector_store = FAISS.load_local(
    "/home/tamil/work/Dohts/faiss_index", embeddings, allow_dangerous_deserialization=True
)


# llm = init_chat_model("gpt-4o-mini", model_provider="openai")

llm = ChatOllama(
    model="qwq:latest",
    base_url=os.getenv('base_url'), 
    client_kwargs={'headers': headers},
    temperature=0,
)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = new_vector_store.similarity_search(state["question"], search_kwargs={"k": 4})
    print("In retriever")
    return {"context": retrieved_docs}


def generate(state: State):
    print("In generator")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


print("Compiling")
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
print("Compiled")



print("Invoking")
result = graph.invoke({"question": "What is the company's focus in the future?"})
print("Invoked")

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')