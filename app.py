import os
from dotenv import load_dotenv

# LLM + Prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Chains
from langchain.chains import LLMChain, RetrievalQA

# Memory
from langchain.memory import ConversationBufferMemory

# RAG
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Text splitter
from langchain_text_splitters import CharacterTextSplitter


# -------- Load API --------
load_dotenv()

# -------- LLM --------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -------- Memory --------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -------- Load & Split KB --------
loader = TextLoader("knowledge_base.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# -------- Embeddings + Vector DB --------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# -------- RAG Chain --------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -------- Intent Prompt --------
intent_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Classify the intent of the user message into one of these:
- greeting
- pricing
- high_intent
- general

Message: {text}

Return only one word.
"""
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# -------- Lead Capture --------
lead = {"name": None, "email": None, "platform": None, "stage": None}

def mock_lead_capture(name, email, platform):
    return f"✅ Lead captured successfully: {name}, {email}, {platform}"

# -------- Hybrid Intent Function --------
def detect_intent(user_input):
    text = user_input.lower()

    if any(w in text for w in ["hi", "hello", "hey"]):
        return "greeting"

    if any(w in text for w in ["price", "plan", "cost"]):
        return "pricing"

    if any(w in text for w in ["buy", "subscribe", "interested", "pro plan"]):
        return "high_intent"

    result = intent_chain.run(text)
    return result.strip()

# -------- Chat Loop --------
print("🤖 AutoStream AI Agent (Hybrid LLM + Intent)\nType 'exit' to stop\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    intent = detect_intent(user_input)
    response = ""

    if intent == "greeting":
        response = "Hello! 👋 How can I help you today?"

    elif intent == "pricing":
        response = qa_chain.run(user_input)

    elif intent == "high_intent":
        if not lead["name"]:
            lead["stage"] = "name"
            response = "Great! What's your name?"

        elif lead["stage"] == "name":
            lead["name"] = user_input
            lead["stage"] = "email"
            response = "Please enter your email."

        elif lead["stage"] == "email":
            lead["email"] = user_input
            lead["stage"] = "platform"
            response = "Which platform do you use?"

        elif lead["stage"] == "platform":
            lead["platform"] = user_input
            response = mock_lead_capture(
                lead["name"], lead["email"], lead["platform"]
            )
            lead = {"name": None, "email": None, "platform": None, "stage": None}

    else:
        response = qa_chain.run(user_input)

    print("Bot:", response)