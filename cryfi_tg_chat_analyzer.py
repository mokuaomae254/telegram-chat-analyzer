import sys
print("Python executable:", sys.executable)
# The rest of your code starts here, e.g., from telethon.sync import TelegramClient
import streamlit as st
from telethon import TelegramClient # Removed .sync as we're handling async explicitly
from telethon.tl.types import MessageMediaPoll
from datetime import datetime, timedelta
import asyncio
import json
import base64
import os

# --- Configuration for Telegram API ---
# Get API ID and API Hash from Streamlit secrets
# You need to create a .streamlit/secrets.toml file with:
# TELEGRAM_API_ID = "YOUR_API_ID"
# TELEGRAM_API_HASH = "YOUR_API_HASH"
try:
    api_id = st.secrets["TELEGRAM_API_ID"]
    api_hash = st.secrets["TELEGRAM_API_HASH"]
except KeyError:
    st.error("Telegram API credentials not found in Streamlit secrets.")
    st.info("Please add TELEGRAM_API_ID and TELEGRAM_API_HASH to your .streamlit/secrets.toml file.")
    st.stop()

# Session name for Telethon client. This will correspond to a file like 'telegram_summarizer_session.session'
# You need to ensure this file is generated locally and then pushed to your GitHub repo.
session_name = 'telegram_summarizer_session'

# --- Gemini API Configuration ---
# The API key is automatically provided by the Canvas environment if left empty.
GEMINI_API_KEY = "" # Leave this empty for Canvas environment

# --- Helper function to call Gemini API ---
async def call_gemini_api(prompt, schema=None):
    """
    Calls the Gemini API to generate text based on the given prompt.
    Optionally, can request a structured JSON response using a schema.
    """
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json" if schema else "text/plain",
            "responseSchema": schema
        } if schema else {}
    }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        # Use aiohttp for async HTTP requests for better compatibility in async contexts
        # You'll need to add aiohttp to your requirements.txt
        # pip install aiohttp
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers={'Content-Type': 'application/json'}, json=payload) as response:
                result = await response.json()

        if response.status != 200:
            st.error(f"Gemini API Error: {result.get('error', {}).get('message', 'Unknown error')}")
            return None

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            if schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    st.warning("Gemini API returned invalid JSON. Trying raw text.")
                    return text_response
            return text_response
        else:
            st.warning("Gemini API response structure unexpected or content missing.")
            return None
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- Function to get Telegram client ---
@st.cache_resource
def get_telegram_client(api_id, api_hash, session_name):
    """Initializes, connects, and ensures authorization of the Telegram client."""

    async def _init_and_connect():
        # Telethon client initialization is safe here
        client = TelegramClient(session_name, api_id, api_hash)
        
        # Connect to Telegram
        await client.connect()

        # Check if authorized and handle session.
        # For Streamlit Cloud, you MUST pre-generate the session file locally
        # and upload it to your GitHub repo. Interactive login is not feasible.
        if not await client.is_user_authorized():
            st.error("Telegram client is not authorized. "
                     "Please ensure your session file (.session) was generated locally (by running client.start() once) "
                     "and then pushed to your GitHub repository.")
            st.stop() # Stop the app if authorization fails
        
        return client

    try:
        # Run the async initialization and connection in a dedicated event loop
        # This is where the core fix for the RuntimeError is.
        return asyncio.run(_init_and_connect())
    except Exception as e:
        st.error(f"Failed to initialize or connect to Telegram client. Error: {e}")
        st.info("Make sure your API ID/Hash are correct and your .session file is valid and present in the repo.")
        st.stop()


# --- Function to fetch messages ---
async def fetch_telegram_messages(client, group_entity, start_date, end_date):
    """
    Fetches messages from a given Telegram group within a specified date range.
    """
    messages = []
    st.write(f"Fetching messages from '{group_entity.title}'...")
    message_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Iterate through messages, filtering by date
        # Adjusted offset_date and reverse for efficiency with date range
        async for message in client.iter_messages(group_entity, offset_date=end_date + timedelta(days=1), reverse=True):
            if message.date < start_date:
                break # Stop if messages are older than the start date
            if message.date >= start_date and message.date <= end_date + timedelta(days=1):
                messages.append(message)
                message_count += 1
                if message_count % 100 == 0: # Update progress every 100 messages
                    status_text.text(f"Fetched {message_count} messages...")
                    # Update progress bar (limited to 100 for display)
                    progress_bar.progress(min(message_count / 1000, 1.0)) # Assuming max 1000 messages for progress bar
        status_text.text(f"Finished fetching. Total messages: {message_count}")
        progress_bar.progress(1.0)
    except Exception as e:
        st.error(f"Error fetching messages: {e}")
        return []
    return messages

# --- Main Streamlit App Layout ---
st.set_page_config(page_title="Telegram Group Chat Summarizer", layout="wide")

st.title("💬 Telegram Group Chat Summarizer")
st.markdown("""
This application helps you download Telegram group chats for a specific timeframe and then provides AI-generated summaries,
with a focus on quizzes and potential scam activities.
""")

# --- Telegram Client Initialization and Connection ---
# This call is wrapped by st.cache_resource, so it runs only once per session
# and handles the async initialization internally.
client = get_telegram_client(api_id, api_hash, session_name)
st.success("Telegram Client initialized and connected!") # Indicate success after get_telegram_client finishes

# --- User Inputs ---
st.header("1. Enter Group Details and Timeframe")

group_input = st.text_input(
    "Telegram Group Link or ID (e.g., `https://t.me/your_group_name` or `-1001234567890`)",
    placeholder="https://t.me/example_group"
)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", datetime.now().date())

# Ensure end_date is not before start_date
if start_date > end_date:
    st.error("End date cannot be before start date.")
    st.stop()

# --- Process Request Button ---
if st.button("Process Group Chat", type="primary"):
    if not group_input:
        st.warning("Please enter a Telegram group link or ID.")
        st.stop()

    group_entity = None
    with st.spinner(f"Resolving group '{group_input}'..."):
        try:
            # All calls to telethon methods must be awaited, and wrapped in asyncio.run()
            # if called from synchronous Streamlit code.
            group_entity = asyncio.run(client.get_entity(group_input))
            st.success(f"Found group: **{group_entity.title}**")
        except Exception as e:
            st.error(f"Could not find group. Please check the link/ID. Error: {e}")
            st.stop()

    if group_entity:
        with st.spinner("Fetching messages... This might take a while for large groups."):
            # Convert dates to datetime objects for comparison with message.date
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            messages = asyncio.run(fetch_telegram_messages(client, group_entity, start_datetime, end_datetime))

        st.success(f"Successfully fetched {len(messages)} messages.")

        if not messages:
            st.info("No messages found in the specified timeframe.")
            st.stop()

        # --- Filter messages for quizzes and potential scammers ---
        quiz_messages = []
        scam_messages = []
        
        # Keywords for quizzes (case-insensitive)
        quiz_keywords = ["quiz", "question", "poll", "answer", "test", "challenge", "score", "correct", "wrong"]
        
        # Keywords for potential scammers (case-insensitive)
        scam_keywords = [
            "invest", "profit", "guaranteed returns", "crypto", "forex", "dm me", 
            "private message", "send money", "urgent", "opportunity", "link", 
            "click here", "join now", "giveaway", "free money", "winning", 
            "lucky draw", "account verification", "wallet", "mining", "signal group",
            "binance", "trading", "deposit", "withdrawal", "payout", "investment platform"
        ]

        for msg in messages:
            text_content = msg.message.lower() if msg.message else ""
            
            # Check for quizzes
            if isinstance(msg.media, MessageMediaPoll):
                quiz_messages.append(msg)
            elif any(keyword in text_content for keyword in quiz_keywords):
                quiz_messages.append(msg)
            
            # Check for scammers
            if any(keyword in text_content for keyword in scam_keywords):
                scam_messages.append(msg)
            # Add more sophisticated checks if needed, e.g., URL patterns, specific user mentions

        st.header("2. AI-Powered Summaries")

        # --- Summarize Quizzes ---
        st.subheader("📊 Quiz Activity Summary")
        if quiz_messages:
            # Ensure sender's first name is included in the text for the LLM
            quiz_texts = "\n---\n".join([f"Date: {m.date}\nSender: {m.sender.first_name if m.sender else 'Unknown'}\nMessage: {m.message}" for m in quiz_messages if m.message])
            if quiz_texts:
                quiz_prompt = f"""Generate a professional summary of the following Telegram group chat messages related to quizzes.
                For each quiz or significant question, **explicitly mention the name of the user who posted it**.
                Focus on the main topics of the quizzes, any notable questions or answers, and the general engagement level.
                Identify if there were any specific winners or participants mentioned by name.
                Keep the summary concise, professional, and informative.

                Messages:
                {quiz_texts}
                """
                with st.spinner("Generating quiz summary..."):
                    quiz_summary = asyncio.run(call_gemini_api(quiz_prompt))
                    if quiz_summary:
                        st.markdown(quiz_summary)
                    else:
                        st.warning("Could not generate quiz summary.")
            else:
                st.info("No text content found in identified quiz messages to summarize.")
        else:
            st.info("No quiz-related activities found in the specified timeframe.")

        # --- Summarize Scammers ---
        st.subheader("🚨 Potential Scammer Activity Summary")
        if scam_messages:
            # Ensure sender's first name is included in the text for the LLM
            scam_texts = "\n---\n".join([f"Date: {m.date}\nSender: {m.sender.first_name if m.sender else 'Unknown'}\nMessage: {m.message}" for m in scam_messages if m.message])
            if scam_texts:
                scam_prompt = f"""Analyze the following Telegram group chat messages that have been flagged as potentially related to scams.
                Generate a professional summary. For each detected scam attempt, **explicitly mention the name of the user who posted the suspicious message**.
                Summarize the nature of the scam attempts, the methods used by the suspected scammers (e.g., asking for DMs, promising returns, linking to external sites), and any specific keywords or patterns observed.
                Highlight any user interactions with these scam attempts. Be direct, professional, and focus on clearly identifying the scamming activities and the individuals involved.

                Messages:
                {scam_texts}
                """
                with st.spinner("Generating scam activity summary..."):
                    scam_summary = asyncio.run(call_gemini_api(scam_prompt))
                    if scam_summary:
                        st.markdown(scam_summary)
                    else:
                        st.warning("Could not generate scam activity summary.")
            else:
                st.info("No text content found in identified scam messages to summarize.")
        else:
            st.info("No potential scammer activities found in the specified timeframe.")

        # --- Optional: Display Raw Messages (for debugging/review) ---
        st.header("3. Raw Messages (Optional)")
        if st.checkbox("Show all fetched messages"):
            for msg in messages:
                st.json({
                    "date": str(msg.date),
                    "sender_id": msg.sender_id,
                    "sender_name": msg.sender.first_name if msg.sender else "Unknown",
                    "message": msg.message,
                    "is_quiz": isinstance(msg.media, MessageMediaPoll) or any(k in (msg.message.lower() if msg.message else "") for k in quiz_keywords),
                    "is_scam_flagged": any(k in (msg.message.lower() if msg.message else "") for k in scam_keywords)
                })
        
        if st.checkbox("Show only quiz messages"):
            for msg in quiz_messages:
                st.json({
                    "date": str(msg.date),
                    "sender_name": msg.sender.first_name if msg.sender else "Unknown",
                    "message": msg.message,
                })
        
        if st.checkbox("Show only scam flagged messages"):
            for msg in scam_messages:
                st.json({
                    "date": str(msg.date),
                    "sender_name": msg.sender.first_name if msg.sender else "Unknown",
                    "message": msg.message,
                })updatin

#streamlit run cryfi_tg_chat_analyzer.py