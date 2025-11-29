"""
Clean, improved, and production-ready version of `agent_setup.py` for the Support-Agent
Streamlit app. This keeps the same high-level behaviour but with:
- Better structure and logging
- Safe environment handling
- Lazy/robust Supabase initialization
- Single shared embedding instance
- Clear tool definitions and error handling
- Comments explaining each section

Instructions:
- Replace your repo's agent_setup.py with this file (or copy it in).
- Set SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY in Streamlit Secrets or environment.
- Commit & push -> Streamlit will redeploy.

NOTE: This file depends on the same Python packages we prepared in requirements.txt.
"""

import os
import sys
import datetime
import logging
from typing import List

from dotenv import load_dotenv

# LangChain / LLM / Supabase imports
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_executor import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Supabase Python client
from supabase.client import create_client, Client

# ---------- Configuration & Logging ----------
load_dotenv()  # safe: will be ignored on Streamlit Cloud if secrets are used

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent_setup")

# ---------- Environment (required) ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_KEY:
    logger.error("GEMINI_API_KEY is not set. The agent cannot call the LLM without it.")

# Keep GOOGLE_API_KEY for backwards compatibility with google packages that expect it
if GEMINI_KEY:
    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_KEY)

# ---------- Supabase client (lazy + robust) ----------
_SUPABASE_CLIENT: Client | None = None


def get_supabase_client() -> Client:
    """Return a singleton Supabase client. Raises RuntimeError if credentials are missing."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error("Supabase credentials not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY).")
        raise RuntimeError("Supabase credentials missing")

    try:
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully.")
        return _SUPABASE_CLIENT
    except Exception as e:
        logger.exception("Failed to initialize Supabase client: %s", e)
        raise


# ---------- Embeddings (single shared instance) ----------
_EMBEDDINGS = None


def get_embeddings():
    """Return a shared embeddings object for the Gemini models."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    try:
        _EMBEDDINGS = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query",
        )
        logger.info("Embeddings initialized.")
        return _EMBEDDINGS
    except Exception:
        logger.exception("Failed to initialize embeddings. Ensure langchain-google-genai is installed and the API key is set.")
        raise


# ---------- Tools (RAG + Escalation) ----------
@tool
def support_faq_solver(query: str) -> str:
    """Search the knowledge base (Supabase) for relevant documents and return a short summary.

    This tool is resilient: if the vector store or DB query fails, it returns a concise error
    string that the agent can act on (e.g., escalate the ticket).
    """
    try:
        embeddings = get_embeddings()
        supabase_client = get_supabase_client()

        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents",
        )

        docs = vector_store.similarity_search(query, k=3)

        if not docs:
            return "No relevant information found in the knowledge base."

        # Return short joined content (the agent prompt will format final response)
        return "\n---\n".join([getattr(d, "page_content", str(d)) for d in docs])

    except Exception as e:
        logger.exception("support_faq_solver failed: %s", e)
        return f"Knowledge base search failed: {e}"


@tool
def create_support_ticket_supabase(user_query: str) -> str:
    """Insert a support ticket into the Supabase `support_tickets` table.

    Returns a friendly confirmation message or a concise error string on failure.
    """
    try:
        supabase_client = get_supabase_client()
        payload = {
            "user_query": user_query,
            "agent_note": "Escalated due to insufficient RAG context.",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

        supabase_client.table("support_tickets").insert(payload).execute()
        return "Your query has been escalated to a human agent."

    except Exception as e:
        logger.exception("Failed to create support ticket: %s", e)
        return f"Escalation failed: {e}"


# ---------- LLM and Agent Creation ----------

def create_llm():
    """Create and return the LLM wrapper for Gemini. Raises if API key missing."""
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=GEMINI_KEY,
        )
        logger.info("LLM (Gemini) client created.")
        return llm
    except Exception:
        logger.exception("Failed to initialize ChatGoogleGenerativeAI. Check your API key and package versions.")
        raise


def build_system_prompt() -> str:
    """Return the system prompt for the agent. Kept separate for readability and easier editing."""
    return (
        "You are Synapse AI (Tier-1 Support Agent).\n\n"
        "Rules:\n"
        "1. ALWAYS call support_faq_solver first.\n"
        "2. If no data found → escalate using create_support_ticket_supabase.\n"
        "3. Always respond politely.\n\n"
        "Knowledge Base for Support:\n\n"
        "1. Password:\n"
        "   Users can reset their password by clicking the 'Forgot Password' link. They will receive an OTP email. After verifying, they can set a new password. If email does not arrive, check spam or contact support.\n\n"
        "2. Account:\n"
        "   Users can view their account details (name, email, billing, subscription, usage logs). If details are incorrect, contact support.\n\n"
        "3. Refund:\n"
        "   Refunds are available within 7 days for annual plans and 3 days for monthly plans, processed within 5–7 business days.\n\n"
        "4. Escalation:\n"
        "   Support tickets have 3 levels. Unresolved issues escalate automatically after 48 hours.\n\n"
        "5. Profile:\n"
        "   Users can update name and phone from Profile page. Email updates require verification.\n\n"
        "6. Account Deletion:\n"
        "   Submit a request via Settings → Delete Account. Deletion is permanent.\n\n"
        "7. Login:\n"
        "   For login issues, clear cache, check internet, try incognito. If unresolved, escalate to Level 2.\n\n"
        "8. Security:\n"
        "   Security alerts for unusual logins. Review activity and change password immediately.\n\n"
        "9. Plans:\n"
        "   Subscription plans: Basic, Pro, Enterprise. Upgrade/downgrade via Billing section.\n\n"
        "10. Payment:\n"
        "    Update card details and retry if payment fails. Failures due to expired cards or insufficient balance.\n"
    )


# Create prompt template once
prompt = ChatPromptTemplate.from_messages([
    ("system", build_system_prompt()),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# Build agent
def build_agent_executor() -> AgentExecutor:
    """Build the agent and return an AgentExecutor. This is safe to call multiple times.

    The LLM and tools are created lazily so that imports and credentials are validated at runtime
    and errors are easier to surface during deployment.
    """
    llm = create_llm()
    tools = [support_faq_solver, create_support_ticket_supabase]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    logger.info("AgentExecutor created.")
    return executor


# Expose a top-level agent_executor variable so `from agent_setup import agent_executor` works
try:
    agent_executor = build_agent_executor()
    logger.info("✅ Agent Executor Ready!")
except Exception as e:
    # Keep an informative message rather than crashing silently; Streamlit logs will show this
    logger.exception("Failed to build agent executor at import time: %s", e)
    # Re-raise to make deployment fail fast so errors are fixed quickly
    raise

# End of file
