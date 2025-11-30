# Support-Agent-
Synapse AI is an intelligent Tier-1 Customer Support Agent powered by Gemini 2.5 Flash, LangChain, Streamlit, and Supabase.
It retrieves answers from a knowledge base using RAG and escalates unresolved issues via ticket creation in Supabase.
 Architecture Diagram
<img width="1536" height="1024" alt="ChatGPT Image Nov 29, 2025, 05_20_38 PM" src="https://github.com/user-attachments/assets/23cc2761-5c91-4d41-bce6-479a883c7323" />




🧩 Overview

Synapse AI is a support automation system designed to function as a Tier-1 customer support assistant.
It understands user queries, retrieves answers using RAG (Retrieval-Augmented Generation), and automatically creates support tickets for difficult issues.

It is designed for:

Customer support automation

AI agents that read knowledge base articles

Startups needing automated help desks

Projects demonstrating RAG + Agents + Supabase

🌟 Features
🔍 Intelligent FAQ Retrieval (RAG)

Retrieves top 3 relevant answers from Supabase Vector DB

Uses Google Gemini embeddings

🤖 Tier-1 AI Support Agent

Uses Gemini 2.5 Flash through LangChain

Tool-calling agent always attempts RAG first

🆘 Automatic Escalation

If the query has low RAG relevance:

A ticket is created in Supabase: support_tickets table

User gets an escalation confirmation

💬 Streamlit Chat UI

Clean, modern chat interface

Persistent session history

📁 Modular Code

app.py – UI

agent_setup.py – Agent + tools + Supabase connection

support_docs/ – RAG knowledge base

⚠️ Limitations

No authentication (public app)

Limited to provided RAG documents

Supabase free tier may throttle heavy usage

Gemini Flash sometimes gives generic answers if RAG fails

Cannot handle advanced multi-step reasoning beyond Tier-1 queries

🧰 Tech Stack
Frontend

Streamlit

LLM

Google Gemini 2.5 Flash

Google Generative AI Embeddings

Framework

LangChain Tool Calling Agents

Backend (Database)

Supabase

VectorStore (embedding search)

Postgres table (support_tickets)

Other

Python 3.10+

dotenv for secrets loading

🛠 Setup & Run Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Add Environment Variables

Create .env file:

SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
GEMINI_API_KEY=your_gemini_api_key


Or use Streamlit → Settings → Secrets for deployment.

4️⃣ Run Locally
streamlit run app.py

5️⃣ Deploy to Streamlit Cloud

Push repo to GitHub

Go to: https://share.streamlit.io

Select repo & choose app.py

Add secrets:

SUPABASE_URL="..."
SUPABASE_SERVICE_KEY="..."
GEMINI_API_KEY="..."


Deploy 🚀

🚀 Potential Improvements
🔮 Future Enhancements

Add user authentication

Add file upload support → user uploads screenshots

Add email notifications for new tickets

Add Sentiment analysis for prioritization

Add admin dashboard for support agents

Add analytics dashboard in Streamlit

Add voice-based support

Add multilingual support
Picture of AI Support Assistant
<img width="1920" height="1080" alt="Screenshot (116)" src="https://github.com/user-attachments/assets/5f6af3a6-31ed-4a5a-b1a9-48e8af305be1" />

