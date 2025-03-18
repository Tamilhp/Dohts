from dohts.knowledge_directory_extractor import KnowledgeDirectoryExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import base64
import os
from dotenv import load_dotenv
from pathlib import Path



import logging
import warnings


warnings.filterwarnings("ignore")
logging.basicConfig(level="INFO")


class IngestData:
    def __init__(self, knowledge_directory: str = None, embedding_model: str = None, vector_db: str = None):
        dotenv_path = Path('/home/tamil/work/Dohts/src/dohts/creds.env')
        load_dotenv(dotenv_path=dotenv_path)
        self.embedding_model = None
        self.vector_store = None
        self.knowledge_directory = knowledge_directory
        self.vector_db = vector_db
        self._select_embedding_model(embedding_model=embedding_model)
        self._select_vector_db(self.vector_db, self.embedding_model)
    
    def __call__(self):
        logging.info("Pipeline Called")

        # calling the knowledge directory loader
        knowledge_docs = KnowledgeDirectoryExtractor.directory_loader(self.knowledge_directory)
        self.create_chunks(knowledge_docs[:2])
    
    
    def _select_embedding_model(self, embedding_model) -> None:
        logging.info("Selecting Embedding Model")
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
        print(os.getenv('base_url'))
        
        self.embedding_model = OllamaEmbeddings(model=embedding_model, base_url="https://ollama.own1.aganitha.ai", 
                                      client_kwargs={'headers': headers})
    
    def _select_vector_db(self, vectordb, embedding_model) -> None:
        logging.info("Selecting vector_db")
        if vectordb == "FAISS":
            from langchain_community.vectorstores import FAISS

            # Initialize the FAISS index
            faiss_index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("hello world")))

            # Initialize the FAISS class
            self.vector_store = FAISS(embedding_function=self.embedding_model, 
                                      index=faiss_index, 
                                      docstore=InMemoryDocstore(), 
                                      index_to_docstore_id={})


    def create_chunks(self, docs) -> None:
        logging.info("Chunks are being created")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        self.index_chunks(self.vector_store, all_splits)
    
    def index_chunks(self, vector_store, all_splits):
        logging.info("Indexing the Chunks")
        # print(all_splits[0])
        vector_store.add_documents(documents=all_splits)

        # save the vectore store
        vector_store.save_local("faiss_index")



        


    