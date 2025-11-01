import streamlit as st
import os
import shutil
import pandas as pd
from datetime import datetime
# --- New Import for secure environment loading ---
from dotenv import load_dotenv
# -------------------------------------------------
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 🚨 CRITICAL: Load environment variables from .env file
load_dotenv()

# --- 1. Configuration Constants (UPDATED FOR GROQ) ---
# Define the paths relative to the script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(CURRENT_DIR, "knowledge_base")
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "faiss_index") 

# --- NEW: Logging CSV Path (For Interactive Chat) ---
LOG_FILE_PATH = os.path.join(CURRENT_DIR, "interactive_chat_log.csv")
# --- NEW: Logging CSV Path (For Batch Processing - APPEND MODE) ---
BATCH_LOG_FILE_PATH = os.path.join(CURRENT_DIR, "batch_query_log.csv")
# ------------------------------------------------------------------

# Knowledge Base File Name
KNOWLEDGE_FILE_NAME = "healthcare_data.txt"
KNOWLEDGE_BASE_PATH = os.path.join(KNOWLEDGE_BASE_DIR, KNOWLEDGE_FILE_NAME)

# Splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM and Embedding Model Setup
# 🚨 CRITICAL CHANGE: Retrieving the key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Groq-optimized model for speed
LLM_MODEL = "llama-3.1-8b-instant" 
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. Streamlit Caching/Loading Functions ---

@st.cache_resource
def load_embeddings():
    """Loads the HuggingFace embeddings model (Runs locally, no HF token needed)."""
    print(f"Loading embeddings model: {EMBEDDINGS_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return embeddings

@st.cache_resource
def load_llm(token, model_name):
    """Loads the Groq LLM using the Groq API Key."""
    if not token:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
        
    print(f"Loading LLM: {model_name} via Groq")
    
    # Ensure the Groq token is set in the environment (redundant but safe)
    os.environ["GROQ_API_KEY"] = token 
    
    try:
        # ChatGroq automatically reads GROQ_API_KEY from os.environ
        llm = ChatGroq(
            model_name=model_name, 
            temperature=0.5, 
            # The token is read automatically from the environment
        )
        return llm
    except Exception as e:
        # Re-raise the error for the main function to catch and display
        raise ValueError(f"Failed to initialize Groq LLM. Details: {e}")

def delete_vector_db():
    """Utility function to clear the FAISS index directory."""
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        st.cache_resource.clear()
        st.success("FAISS index and Streamlit cache cleared successfully. Please rerun the app to rebuild.")
        st.experimental_rerun()
    else:
        st.warning("FAISS index directory not found. No files deleted.")

@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_and_index_documents(knowledge_file_path, vector_db_path, _embeddings):
    """
    Loads text document, handles encoding errors, splits text, and creates/updates a FAISS vector store.
    """
    
    # 🚨 CRITICAL: Check if the knowledge base directory and file exist
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        st.error(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        return None
        
    if not os.path.exists(knowledge_file_path):
        print(f"Knowledge file not found: {knowledge_file_path}")
        st.error(f"Required file not found: **{KNOWLEDGE_FILE_NAME}**. Please place it in the '{os.path.basename(KNOWLEDGE_BASE_DIR)}' folder.")
        return None

    # --- Index Loading/Recreation Logic ---
    index_exists = os.path.exists(vector_db_path)
    vector_store = None

    if index_exists:
        # Load existing index if available
        print(f"Attempting to load existing FAISS index from {vector_db_path}...")
        try:
            vector_store = FAISS.load_local(vector_db_path, _embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load FAISS index ({e}). Index files might be missing or corrupted. Forcing recreation.")
            if os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)
                print(f"Cleaned up corrupted index directory: {vector_db_path}")
            index_exists = False 

    if not index_exists:
        # --- File Loading and Splitting (Only run if index needs creation) ---
        all_documents = []
        try:
            # Try standard UTF-8 encoding first
            loader = TextLoader(knowledge_file_path, encoding='utf-8')
            documents = loader.load()
        except Exception as e_utf8:
            # If UTF-8 fails, try Latin-1 fallback
            print(f"Warning: UTF-8 failed for {KNOWLEDGE_FILE_NAME}. Trying latin-1. Error: {e_utf8}")
            try:
                loader = TextLoader(knowledge_file_path, encoding='latin-1')
                documents = loader.load()
            except Exception as e_latin1:
                print(f"ERROR: Failed to load {KNOWLEDGE_FILE_NAME}. Final Error: {e_latin1}")
                st.error(f"Error loading data file. Check file encoding/permissions. Details: {e_latin1}")
                return None
        
        all_documents.extend(documents)
        
        if not all_documents:
            print("ERROR: No content was loaded from the file.")
            st.error("The data file appears empty or content could not be read.")
            return None

        # Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        split_documents = text_splitter.split_documents(all_documents)
        print(f"Split documents into {len(split_documents)} chunks.")

        # Create new index
        print(f"Creating new FAISS index at {vector_db_path}...")
        vector_store = FAISS.from_documents(split_documents, _embeddings)
        vector_store.save_local(vector_db_path)
        print("FAISS index saved successfully.")
    
    return vector_store

# --- 3. Retrieval Chain Setup ---

def setup_retrieval_qa_chain(llm, vector_store):
    """Sets up the RetrievalQA chain with a custom prompt."""
    
    # Multilingual Prompt Template
    template = """You are a helpful and multilingual AI assistant. 
    Use the following pieces of context to answer the user's question. 
    Answer in the same language as the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# --- Function to append single query result to log CSV (Interactive Chat) ---
def append_to_chat_log(query, response, source_docs, status="Success"):
    """Appends a single query and its result to the interactive chat log CSV."""
    
    # Format source documents for logging
    sources_combined = ' | '.join([
        f"Source {j+1}: {doc.page_content[:200]}..." for j, doc in enumerate(source_docs)
    ])
    source_metadata = ' | '.join([
        doc.metadata.get('source', 'N/A') for doc in source_docs
    ])
    
    new_record = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Query": query,
        "Answer": response,
        "Source_Metadata": source_metadata,
        "Source_Content_Snippets": sources_combined,
        "Status": status
    }])
    
    # Check if file exists to determine if header should be written
    write_header = not os.path.exists(LOG_FILE_PATH)
    
    try:
        new_record.to_csv(
            LOG_FILE_PATH, 
            mode='a', 
            header=write_header, 
            index=False, 
            encoding='utf-8'
        )
        print(f"Logged interactive query to {LOG_FILE_PATH}")
    except Exception as e:
        print(f"WARNING: Failed to append to chat log CSV: {e}")
# -------------------------------------------------------------

# --- Function to handle batch processing and CSV generation (MODIFIED FOR SINGLE FILE APPEND) ---
def process_batch_queries(qa_chain, batch_queries):
    """
    Processes a list of queries, returns results as a DataFrame, 
    and APPENDS the results to a single, cumulative CSV log file.
    """
    
    results = []
    
    if not batch_queries:
        st.warning("No queries entered for batch processing.")
        return None
    
    # Clean up and filter out empty lines
    queries = [q.strip() for q in batch_queries.split('\n') if q.strip()]

    if not queries:
        st.warning("The batch query input contains no valid queries.")
        return None
        
    st.info(f"Processing {len(queries)} queries...")
    
    # Create a placeholder for status updates
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, query in enumerate(queries):
        progress = (i + 1) / len(queries)
        progress_bar.progress(progress)
        status_text.text(f"Processing query {i+1} of {len(queries)}: '{query[:50]}...'")

        try:
            # Run the QA chain
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            # Format source documents into a single string
            sources_combined = ' | '.join([
                f"Source {j+1}: {doc.page_content[:200]}..." for j, doc in enumerate(source_docs)
            ])
            source_metadata = ' | '.join([
                doc.metadata.get('source', 'N/A') for doc in source_docs
            ])

            results.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # ADD TIMESTAMP HERE
                "Query": query,
                "Answer": answer,
                "Source_Metadata": source_metadata,
                "Source_Content_Snippets": sources_combined,
                "Status": "Success"
            })
            
        except Exception as e:
            results.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # ADD TIMESTAMP HERE
                "Query": query,
                "Answer": f"ERROR: {e}",
                "Source_Metadata": "N/A",
                "Source_Content_Snippets": "N/A",
                "Status": "Error"
            })
            st.error(f"Error processing query '{query[:30]}...': {e}")
            
    progress_bar.empty()
    status_text.success(f"Batch processing complete! {len(queries)} queries processed.")
    
    df_results = pd.DataFrame(results)

    # 🚨 CRITICAL MODIFICATION: Append to a single file (batch_query_log.csv)
    output_path = BATCH_LOG_FILE_PATH # Use the newly defined single log path
    write_header = not os.path.exists(output_path) # Check if file exists to write header

    try:
        df_results.to_csv(
            output_path, 
            index=False, 
            encoding='utf-8',
            mode='a', # CRITICAL: Append mode
            header=write_header # Write header only if the file is new
        )
        st.success(f"Batch results appended successfully to: **{os.path.basename(output_path)}**")
        print(f"Appended batch results to: {output_path}")
    except Exception as e:
        st.error(f"Failed to append batch results to disk: {e}")
        print(f"ERROR: Failed to append batch results to disk: {e}")

    return df_results
# -------------------------------------------------------------------


# --- 4. Streamlit App Logic (MODIFIED) ---

def main():
    st.set_page_config(page_title="Groq-Accelerated RAG Chatbot 💬", layout="wide") 

    st.title("Groq-Accelerated RAG Chatbot 💬")
    st.markdown("Ask questions about the healthcare data in any language! (Powered by Groq)")

    # 1. Initialize Components
    try:
        embeddings = load_embeddings()
        llm = load_llm(GROQ_API_KEY, LLM_MODEL)
    except Exception as e:
        st.error(f"Error loading LLM or Embeddings. Details: {e}")
        if "GROQ_API_KEY not found" in str(e):
             st.markdown("**Please ensure you have a `.env` file with `GROQ_API_KEY='gsk_...'` in the same directory.**")
        return

    # 2. Load Vector Store
    vector_store = load_and_index_documents(KNOWLEDGE_BASE_PATH, VECTOR_DB_PATH, embeddings)

    if vector_store is None:
        st.stop() 

    # 3. Setup QA Chain
    qa_chain = setup_retrieval_qa_chain(llm, vector_store)
    
    # --- Sidebar for Index Management and Batch Query (MODIFIED) ---
    with st.sidebar:
        st.header("Index Management")
        if st.button("🚨 Rebuild FAISS Index", type="primary", help="Deletes the existing FAISS folder and forces a complete rebuild from source data."):
            delete_vector_db()

        st.markdown("---")
        # 🚨 MODIFICATION: Updated info box for both log files
        st.info(f"Interactive chat log: **{os.path.basename(LOG_FILE_PATH)}**")
        st.info(f"Batch results log: **{os.path.basename(BATCH_LOG_FILE_PATH)}**")
        st.markdown("---")
        
        st.header("Batch Query Mode 📊")
        batch_query_input = st.text_area(
            "Enter Queries (One per Line):", 
            key="batch_input",
            height=200,
            placeholder="What is the definition of Type 2 Diabetes?\nHow do clinical trials work?\nWhich region has the highest rate of cancer?"
        )
        
        # 🚨 MODIFICATION: Updated button text and logic to reflect saving to disk
        if st.button("Run Batch & Append to Log", key="run_batch"):
            with st.spinner("Running batch queries and appending results to log..."):
                # The saving logic is now inside process_batch_queries
                df_results = process_batch_queries(qa_chain, batch_query_input)
                
            if df_results is not None:
                st.subheader("Batch Results Summary")
                st.dataframe(df_results)

    # 4. Interactive Chat Interface 
    
    st.subheader("Interactive Chat Mode")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response
        with st.chat_message("assistant"):
            response = ""
            source_docs = []
            status = "Error" # Default status
            
            with st.spinner("Thinking... (Groq is fast!)"):
                try:
                    # Run the QA chain
                    result = qa_chain.invoke({"query": prompt}) 
                    response = result["result"]
                    source_docs = result.get("source_documents", [])
                    status = "Success" # Update status on successful completion
                    
                    st.markdown(response)
                    
                    # Display sources for transparency
                    if source_docs:
                        with st.expander("Source Documents"):
                            for i, doc in enumerate(source_docs[:3]): # Show top 3 sources
                                st.text_area(f"Source {i+1} (Metadata: {doc.metadata.get('source', 'N/A')})", doc.page_content, height=100)

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print("FULL ERROR TRACE:\n", error_details)
                    response = f"An error occurred during query processing: {e}"
                    st.error(response)
                    status = "Failure" # Mark as failure

            # 🚨 CRITICAL STEP: Log the query and result to disk automatically
            append_to_chat_log(prompt, response, source_docs, status)
            
            # Add the final response to the session state
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    # Create the knowledge base directory if it doesn't exist
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        print(f"Created knowledge base directory: {KNOWLEDGE_BASE_DIR}")
        
    main()