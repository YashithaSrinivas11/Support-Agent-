# Cell 3: Agent Logic and Orchestration

import os
import sys
# TOP of agent_core.py (or app.py)

import os
from dotenv import load_dotenv

# CRITICAL: Load the environment file first!
load_dotenv() 

import os
import sys
import datetime
from dotenv import load_dotenv
from supabase.client import create_client, Client

# --- LangChain Imports (CRITICAL FIXES) ---
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
# FIX 1: Import AgentExecutor from the dedicated sub-module
from langchain.agents.agent_executor import AgentExecutor 
# FIX 2: Import the creation function
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# --- 1. Load Environment Variables (SAFE METHOD) ---
# Note: On Streamlit Cloud, this load_dotenv() call may be skipped, 
# but the os.environ.get() calls below retrieve the secrets set in the dashboard.
load_dotenv() 

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# Set the Gemini key as a general environment variable for compatibility
os.environ["GOOGLE_API_KEY"] = GEMINI_KEY


# --- 2. Supabase Client Initialization ---
try:
    # Use the retrieved variables
    SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    # On Streamlit Cloud, this will catch connection errors
    print(f"FATAL: Could not initialize Supabase Client: {e}")
    sys.exit(1)


# ---------------- TOOLS ---------------- #

@tool
def support_faq_solver(query: str) -> str:
    """RAG tool for FAQ retrieval."""
    try:
        # Note: API key is not needed here as it's set globally/in environment
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=SUPABASE_CLIENT,
            table_name="documents",
            query_name="match_documents"
        )
        docs = vector_store.similarity_search(query, k=3)

        if not docs:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join([doc.page_content for doc in docs])

    except Exception as e:
        # Return a concise error to the LLM for escalation decision
        return f"Knowledge base search failed: {e}"


@tool
def create_support_ticket_supabase(user_query: str) -> str:
    """Escalates complex queries to Supabase."""
    payload = {
        "user_query": user_query,
        "agent_note": "Escalated due to insufficient RAG context.",
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        # Insert payload into the support_tickets table
        SUPABASE_CLIENT.table("support_tickets").insert(payload).execute()
        return "Your query has been escalated to a human agent."
    except Exception as e:
        return f"Escalation failed: {e}"


# ---------------- LLM & AGENT ---------------- #

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GEMINI_KEY # Pass the retrieved key for clarity
)

system_prompt = """You are Synapse AI (Tier-1 Support Agent).

Rules:
1. ALWAYS call support_faq_solver first.
2. If no data found → escalate using create_support_ticket_supabase.
3. Always respond politely.

Knowledge Base for Support:

1. Password:
   Users can reset their password by clicking the 'Forgot Password' link. They will receive an OTP email. After verifying, they can set a new password. If email does not arrive, check spam or contact support.

2. Account:
   Users can view their account details (name, email, billing, subscription, usage logs). If details are incorrect, contact support.

3. Refund:
   Refunds are available within 7 days for annual plans and 3 days for monthly plans, processed within 5–7 business days.

4. Escalation:
   Support tickets have 3 levels. Unresolved issues escalate automatically after 48 hours.

5. Profile:
   Users can update name and phone from Profile page. Email updates require verification.

6. Account Deletion:
   Submit a request via Settings → Delete Account. Deletion is permanent.

7. Login:
   For login issues, clear cache, check internet, try incognito. If unresolved, escalate to Level 2.

8. Security:
   Security alerts for unusual logins. Review activity and change password immediately.

9. Plans:
   Subscription plans: Basic, Pro, Enterprise. Upgrade/downgrade via Billing section.

10. Payment:
    Update card details and retry if payment fails. Failures due to expired cards or insufficient balance.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

tools = [support_faq_solver, create_support_ticket_supabase]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


print("✅ Agent Executor Ready!")
