Groq-Accelerated Multilingual Healthcare RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot that leverages Groq LPU acceleration, LangChain, and FAISS vector search for lightning-fast, multilingual healthcare question answering.
The system integrates Groq’s llama-3.1-8b-instant model for ultra-low-latency reasoning, combined with a local knowledge base for factual and explainable responses.

🚀 Key Features

✅ Groq LPU-Powered LLM — Uses llama-3.1-8b-instant via the Groq API for high-speed inference
✅ Multilingual Support — Understands and responds in any language detected in user queries
✅ Retrieval-Augmented Generation (RAG) — Answers are context-grounded using a local healthcare dataset
✅ Persistent FAISS Vector Store — Efficient semantic retrieval of relevant document chunks
✅ Batch Query Processing — Run and log multiple queries automatically to CSV
✅ Interactive Chat Mode — Natural conversation interface using Streamlit’s st.chat_message
✅ Automatic Logging — Separate logs for chat and batch sessions stored as CSV files
✅ Rebuildable Index — One-click FAISS reset for fresh knowledge base updates

🗂️ Project Structure
📦 groq-healthcare-rag-chatbot
│
├── knowledge_base/
│   └── healthcare_data.txt              # Your knowledge source for RAG (text file)
│
├── faiss_index/                         # Auto-generated FAISS vector database
│
├── interactive_chat_log.csv             # Logs interactive chat sessions
├── batch_query_log.csv                  # Logs batch query results
│
├── .env                                 # Contains your GROQ_API_KEY
├── requirements.txt                     # Python dependencies
├── app.py                               # Main Streamlit app
└── README.md                            # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your .env File

Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY="gsk_your_actual_groq_api_key_here"

5️⃣ Prepare Knowledge Base

Place your domain-specific text file inside the knowledge_base/ folder:

knowledge_base/
 └── healthcare_data.txt


This file should contain factual healthcare-related information (guidelines, conditions, terms, etc.).

6️⃣ Run the App
streamlit run app.py


Then open the link shown in your terminal (usually http://localhost:8501).

💬 Usage
🔹 Interactive Chat Mode

Type questions in any language:

“What are the symptoms of diabetes?”
“¿Cuáles son las causas del asma?”
“दिल की बीमारियों के प्रमुख कारण क्या हैं?”

The chatbot retrieves the most relevant passages from the knowledge base, then responds concisely and accurately.

🔹 Batch Query Mode

In the sidebar, enter multiple queries (one per line), for example:

What is hypertension?
What are the preventive measures for cancer?
Explain the role of clinical trials in drug development.


Click "Run Batch & Append to Log" to process all queries and save them in batch_query_log.csv.

🧩 Rebuilding the Vector Store

If you update or replace your healthcare_data.txt, rebuild the FAISS index:

Open the app sidebar.

Click 🚨 Rebuild FAISS Index.

The app will recreate embeddings and vector storage automatically.

📊 Logging System
Log Type	File	Description
Interactive Chat	interactive_chat_log.csv	Logs each user-assistant message pair
Batch Queries	batch_query_log.csv	Logs all batch queries with timestamps and source context

Each log includes:

Timestamp

Query and Answer

Source Document Metadata

Snippet of Retrieved Context

Status (Success/Error)

🧠 Technology Stack
Component	Library / API
LLM Backend	Groq API
 (llama-3.1-8b-instant)
Framework	LangChain

Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Store	FAISS

Frontend	Streamlit

Logging	pandas, datetime, CSV storage
Configuration	.env, dotenv
🧰 Requirements File Example

Include this in your requirements.txt if not already generated:

streamlit
langchain
langchain-groq
langchain-community
faiss-cpu
sentence-transformers
python-dotenv
pandas

🧾 License

This project is open-source under the MIT License.
You’re free to use, modify, and distribute it with attribution.

👨‍💻 Author

Himanshu Pabbi
AI & ML Foundation Researcher
🔗 GitHub Profile https://github.com/himanshuPabbi
