# import streamlit as st
# import os
# import logging 
# from langchain.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# import ollama
# #CONFIGGure info
# logging.basicConfig(level=logging.INFO)

# #constanta
# DOC_PATH = "./data/BOI.pdf"
# MODEL_NAME = "llama3.2"

# EMBEDDING_MODEL = "nomic-embed-text"
# VECTOR_STORE_NAME = "simple-rag"
# PERSIST_DIRECTORY = "./chroma_db"

# def ingest_pdf(doc_path):
#     """LOAD PDF document"""
#     if os.path.exists(doc_path):
#         loader = PyPDFLoader(doc_path)
#         data = loader.load()
#         logging.info("PDF loaded successfully.")
#         return data

#     else:
#         logging.error(f"PDF file not found at path: {doc_path}")
#         st.error("PDF file not found.")
#         return None
    


# def split_documents(documents):
#     """Split documents into smaller chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
#     chunks = text_splitter.split_documents(documents)
#     logging.info("Documents split into chunks.")
#     return chunks


# @st.cache_resource
# def load_vector_db():
#     """Load or create the vector database."""
#     # Pull the embedding model if not already available
#     ollama.pull(EMBEDDING_MODEL)

#     embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

#     if os.path.exists(PERSIST_DIRECTORY):
#         vector_db = Chroma(
#             embedding_function=embedding,
#             collection_name=VECTOR_STORE_NAME,
#             persist_directory=PERSIST_DIRECTORY,
#         )
#         logging.info("Loaded existing vector database.")
#     else:
#         # Load and process the PDF document
#         data = ingest_pdf(DOC_PATH)
#         if data is None:
#             return None

#         # Split the documents into chunks
#         chunks = split_documents(data)

#         vector_db = Chroma.from_documents(
#             documents=chunks,
#             embedding=embedding,
#             collection_name=VECTOR_STORE_NAME,
#             persist_directory=PERSIST_DIRECTORY,
#         )
#         vector_db.persist()
#         logging.info("Vector database created and persisted.")
#     return vector_db


# def create_retriever(vector_db, llm):
#     """Create a multi-query retriever."""
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate five
# different versions of the given user question to retrieve relevant documents from
# a vector database. By generating multiple perspectives on the user question, your
# goal is to help the user overcome some of the limitations of the distance-based
# similarity search. Provide these alternative questions separated by newlines.
# Original question: {question}""",
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
#     )
#     logging.info("Retriever created.")
#     return retriever


# def create_chain(retriever, llm):
#     """Create the chain with preserved syntax."""
#     # RAG prompt
#     template = """Answer the question based ONLY on the following context:
# {context}
# Question: {question}
# """

#     prompt = ChatPromptTemplate.from_template(template)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     logging.info("Chain created with preserved syntax.")
#     return chain


# def main():
#     st.title("Document Assistant")

#     # User input
#     user_input = st.text_input("Enter your question:", "")

#     if user_input:
#         with st.spinner("Generating response..."):
#             try:
#                 # Initialize the language model
#                 llm = ChatOllama(model=MODEL_NAME)

#                 # Load the vector database
#                 vector_db = load_vector_db()
#                 if vector_db is None:
#                     st.error("Failed to load or create the vector database.")
#                     return

#                 # Create the retriever
#                 retriever = create_retriever(vector_db, llm)

#                 # Create the chain
#                 chain = create_chain(retriever, llm)

#                 # Get the response
#                 response = chain.invoke(input=user_input)

#                 st.markdown("**Assistant:**")
#                 st.write(response)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#     else:
#         st.info("Please enter a question to get started.")


# if __name__ == "__main__":
#     main()
import streamlit as st
import os
import logging
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import shutil  # For deleting directories

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

def ingest_pdf(doc_path):
    """Load PDF document."""
    loader = PyPDFLoader(doc_path)
    data = loader.load()
    logging.info("PDF loaded successfully.")
    return data

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

@st.cache_resource
def load_vector_db(doc_path=None):
    """Load or create the vector database."""
    # Ensure Ollama model is pulled and available
    try:
        ollama.pull(EMBEDDING_MODEL)  # Pull the embedding model to ensure it's available
    except Exception as e:
        logging.error(f"Error pulling Ollama model: {e}")
        st.error("Failed to connect to Ollama model.")
        return None

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        # If the vector store already exists, load it
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # If the vector store does not exist, process the document and create a new one
        if doc_path is None:
            logging.error("No document path provided to create the vector DB.")
            return None

        # Load and process the uploaded PDF document
        data = ingest_pdf(doc_path)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        # Create vector store from documents
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

def delete_temp_files():
    """Delete temporary files and the vector database."""
    # Delete the uploaded PDF
    if os.path.exists("./temp_uploaded.pdf"):
        os.remove("./temp_uploaded.pdf")
        logging.info("Uploaded PDF deleted.")
    
    # Delete the Chroma vector database
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)  # Delete entire directory
        logging.info("Chroma vector database deleted.")

def main():
    st.title("Document Assistant")

    # PDF file uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    # User input for question
    user_input = st.text_input("Enter your question:", "")

    if uploaded_file is not None and user_input:
        with st.spinner("Processing your document and generating response..."):
            try:
                # Save the uploaded PDF to a temporary path
                temp_pdf_path = "./temp_uploaded.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize the language model
                llm = ChatOllama(model=MODEL_NAME)

                # Load the vector database using the uploaded PDF
                vector_db = load_vector_db(doc_path=temp_pdf_path)
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            finally:
                # Delete temporary files and vector database after processing
                delete_temp_files()

    else:
        st.info("Please upload a PDF and enter a question to get started.")

if __name__ == "__main__":
    main()
