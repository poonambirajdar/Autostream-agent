# 🤖 AutoStream AI Agent (LangChain + RAG + Intent Detection)

An intelligent conversational AI agent built using **LangChain**, **OpenAI**, and **FAISS**, featuring:

* 🔍 Retrieval-Augmented Generation (RAG)
* 🧠 Hybrid Intent Detection (Rule-based + LLM)
* 💬 Conversational Memory
* 📊 Lead Capture Flow
* ⚡ Fast local vector search with FAISS

## 🚀 Features

* **Intent Classification**

  * Greeting
  * Pricing queries
  * High-intent users (lead capture)
  * General queries

* **RAG Pipeline**

  * Loads knowledge base from text file
  * Splits into chunks
  * Embeds using OpenAI
  * Stores in FAISS vector DB

* **Lead Capture System**

  * Collects:
    * Name
    * Email
    * Platform

* **Memory**

  * Maintains conversation history

## 🛠️ Tech Stack

* Python 3.11
* LangChain
* OpenAI API
* FAISS (Vector Database)
* dotenv

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

## ▶️ Run the Project

```bash
python app.py
```

## 💬 Example Usage

```
You: hi
Bot: Hello! 👋 How can I help you today?

You: what is your pricing?
Bot: [retrieved answer from knowledge base]

You: I want to buy
Bot: Great! What's your name?
```

## 📁 Project Structure

```
autostream-agent/
│
├── app.py
├── knowledge_base.txt
├── requirements.txt
├── .env
└── README.md
```

## ⚠️ Important Notes

* Use **Python 3.11** (recommended for compatibility)
* Ensure your OpenAI API key is valid
* FAISS may require proper installation on Windows

## 🔧 Requirements.txt

```txt
langchain==0.1.20
langchain-openai==0.1.7
langchain-community
langchain-text-splitters
openai==1.30.5
faiss-cpu
python-dotenv
```
