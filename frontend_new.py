import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import sqlite3

try:
    from backend import init_database, get_response
except ImportError:
    st.error("Fehler: backend.py konnte nicht gefunden werden. Stellen Sie sicher, dass sich die Datei im selben Verzeichnis befindet.")
    st.stop() 

st.set_page_config(page_title="AngeBOT", page_icon="🛒")
load_dotenv()

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

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Tell me what you want to cook...")
    if user_query is not None and user_query.strip() != "":
    # Neue Benutzereingabe zur Chat-Historie hinzufügen
        st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)
        
    # Prüfen, ob eine Datenbankverbindung besteht
    if st.session_state.db is not None:
        with st.chat_message("AI", avatar="🤖"):
            with st.spinner("Denke nach..."):
                try:
                    # Antwort auf Benutzereingabe holen und anzeigen (über Backend-Funktion)
                    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                    st.markdown(response)
                    # Antwort auch zur Chat-Historie hinzufügen
                    st.session_state.chat_history.append(AIMessage(content=response))
                except Exception as e:
                    error_message = f"Entschuldigung, es ist ein Fehler aufgetreten: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append(AIMessage(content=error_message))
    else:
        # Fehlermeldung, wenn keine Datenbankverbindung besteht
        no_db_message = "Bitte verbinden Sie sich zuerst mit einer Datenbank über die Seitenleiste."
        with st.chat_message("AI", avatar="🤖"):
            st.warning(no_db_message)
        st.session_state.chat_history.append(AIMessage(content=no_db_message))


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