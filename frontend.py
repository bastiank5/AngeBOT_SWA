import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import sqlite3
from backend import get_response, init_database

st.set_page_config(page_title="AngeBOT", page_icon="🛒")
load_dotenv()

# === DATABASE INITIALIZATION ===
# Initialize db using the function from backend.py
# This should be done once, ideally when the session starts.
if "db" not in st.session_state or st.session_state.db is None:
    try:
        st.session_state.db = init_database() # Call init_database here
        # You might want to add a log or a success message here for debugging
        # st.toast("Datenbank erfolgreich initialisiert!")
    except Exception as e:
        st.error(f"Fehler bei der Datenbankinitialisierung: {e}")
        # Stop the app or handle the error as appropriate
        st.stop()

# === SQLite INIT ===
conn = sqlite3.connect("AngeBot.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        name TEXT,
        city TEXT,
        preferences TEXT,
        transport TEXT,
        age INTEGER,
        budget REAL
    )
""")
conn.commit()

st.session_state.db = init_database()
st.success(f"Erfolgreich mit 'AngeBot.db' verbunden!")

# === SESSION INIT ===
if "page" not in st.session_state:
    st.session_state.page = "auth"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"
if "db" not in st.session_state: # Initialize db, assuming it's handled elsewhere
    st.session_state.db = None

# === AUTHENTICATION PAGE ===
def auth_page():
    st.title("AngeBOT Login / Sign-Up")
    if st.button("Create Account"):
        st.session_state.page = "personal_info"

    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        if user:
            st.session_state.current_user = username
            st.session_state.user_info = {
                "name": user[2],
                "city": user[3],
                "preferences": user[4],
                "transport": user[5].split(","),
                "age": user[6],
                "budget": user[7],
            }
            st.success("Login successful.")
            st.session_state.page = "chatbot"
        else:
            st.error("Invalid username or password.")


# === PERSONAL INFORMATION PAGE ===
def personal_info_page():
    st.title("Create Your Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    name = st.text_input("Full Name")
    city = st.text_input("City")
    preferences = st.text_input("Allergies / Preferences")
    transport = st.multiselect("Transport Options", ["Walking", "Bus", "Car", "Bicycle"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    budget = st.number_input("Budget in Euro", min_value=0.0, step=0.5)

    if st.button("Create and Continue to Chatbot"):
        if username and password:
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                st.error("Username already exists.")
            else:
                st.session_state.current_user = username
                st.session_state.user_info = {
                    "name": name,
                    "city": city,
                    "preferences": preferences,
                    "transport": transport,
                    "age": age,
                    "budget": budget
                }
                cursor.execute("""
                    INSERT INTO users (username, password, name, city, preferences, transport, age, budget)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username,
                    password,
                    name,
                    city,
                    preferences,
                    ",".join(transport),
                    age,
                    budget,
                ))
                conn.commit()
                st.session_state.page = "chatbot"
        else:
            st.error("Username and password are required.")


# === CHATBOT PAGE ===
def chatbot_page():
    st.title("AngeBOT: What would you like to cook today?")

    # Sidebar for model selection
    with st.sidebar:
        st.subheader("Model Selection")
        st.session_state.model_name = st.selectbox(
            "Choose AI Model",
            ("gpt-3.5-turbo", "gpt-4"),
            index=("gpt-3.5-turbo", "gpt-4").index(st.session_state.model_name) # set default from session state
        )

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage): # Backward compatibility
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, dict) and message.get("role") == "ai": # New AI message format
            with st.chat_message(f"AI ({message.get('model_name', 'N/A')})"):
                st.markdown(message["content"])
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Tell me what you want to cook...")
    if user_query is not None and user_query.strip() != "":
    # Neue Benutzereingabe zur Chat-Historie hinzufügen
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        # Call get_response with model_name and user_info
        response = get_response(
            user_query,
            st.session_state.db, # Assuming this will be the SQLDatabase object
            st.session_state.chat_history,
            st.session_state.model_name,
            st.session_state.user_info # Pass user_info here
        )

        ai_response_data = {
            "role": "ai",
            "content": response,
            "model_name": st.session_state.model_name
        }
        st.session_state.chat_history.append(ai_response_data)

        with st.chat_message(f"AI ({st.session_state.model_name})"):
            st.markdown(response)


# === DESIGN THEME (Pizza Background) ===
st.markdown("""
    <style>
    html, body, .stApp {
        background: url("https://images.unsplash.com/photo-1506354666786-959d6d497f1a") no-repeat center center fixed;
        background-size: cover;
        color: #f0f0f0;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 2rem;
    }
    .stChatMessage {
        background-color: white !important;
        border-radius: 12px;
        padding: 1rem;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: rgba(255,255,255,0.9);
        color: #000;
        border: 1px solid #ccc;
        font-weight: bold;
    }
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: white;
        font-weight: bold;
    }
    * {
        font-family: 'Comic Sans MS', cursive;
    }
    </style>
""", unsafe_allow_html=True)


# === PAGE ROUTING ===
if st.session_state.page == "auth":
    auth_page()
elif st.session_state.page == "personal_info":
    personal_info_page()
elif st.session_state.page == "chatbot":
    chatbot_page()