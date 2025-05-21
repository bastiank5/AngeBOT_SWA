# frontend.py
# Diese Datei enthält den Streamlit-Code für die Benutzeroberfläche der Chat-Anwendung.
# Technischer Hinweis für den Benutzer:
# Der Quellcode der App kann jederzeit mittels "streamlit run frontend.py" gestartet werden.
# Danach ist die App unter der URL "http://localhost:8501/" erreichbar.
# Stellen Sie sicher, dass auch die Datei backend.py im selben Verzeichnis liegt
# und eine .env Datei mit Ihrem OPENAI_API_KEY existiert.

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage # Wird für die Chat-Historie benötigt
# Importiere die Backend-Funktionen
try:
    from backend import init_database, get_response
except ImportError:
    st.error("Fehler: backend.py konnte nicht gefunden werden. Stellen Sie sicher, dass sich die Datei im selben Verzeichnis befindet.")
    # Beenden Sie die App, wenn das Backend nicht importiert werden kann, um weitere Fehler zu vermeiden.
    st.stop() 

# Grundkonfiguration der Webseite (Titel und Icon)
st.set_page_config(page_title="Chat with SQLite", page_icon=":speech_balloon:")

# Haupttitel auf der Seite
st.title("Chat with SQLite 💬")

# Initialisierung des Session State für die Chat-Historie und die Datenbankverbindung
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hallo! Ich bin ein SQL-Assistent. Wähle eine Datenbank und stelle mir eine Frage."),
    ]
if "db" not in st.session_state:
    st.session_state.db = None # Initial keine Datenbankverbindung

# Aufbau der Seitenleiste (Sidebar)
with st.sidebar:
    st.subheader("Einstellungen")
    st.write("Dies ist eine einfache Chat-Anwendung, die SQLite verwendet. Verbinden Sie sich mit der Datenbank und beginnen Sie mit dem Chatten.")
    
    # Eingabefeld für den Namen der Datenbank und vordefinierten Werten.
    # Der Benutzer gibt hier nur den Namen ein, z.B. "AngeBot"
    db_name_input = st.text_input("Datenbankname (z.B. AngeBot)", value="AngeBot", key="DatabaseName")
    
    # Button, um Verbindung zur Datenbank herzustellen
    if st.button("Verbinden", key="connect_db"):
        if db_name_input:
            with st.spinner(f"Verbinde mit Datenbank '{db_name_input}.db'..."):
                try:
                    # Initialisiert die Datenbankverbindung über die Backend-Funktion
                    # und speichert sie im Session State
                    st.session_state.db = init_database()
                    st.success(f"Erfolgreich mit '{db_name_input}.db' verbunden!")
                    # Optional: Chat zurücksetzen oder eine Bestätigungsnachricht im Chat anzeigen
                    st.session_state.chat_history.append(AIMessage(content=f"Verbunden mit der Datenbank: {db_name_input}.db. Du kannst jetzt Fragen stellen."))
                except Exception as e:
                    st.error(f"Fehler beim Verbinden mit der Datenbank: {e}")
                    st.session_state.db = None # Verbindung zurücksetzen bei Fehler
        else:
            st.warning("Bitte geben Sie einen Datenbanknamen ein.")

# Zeigt bisherige Nachrichten im Chat-Fenster an
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.markdown(message.content)

# Eingabefeld für neue Nachrichten im Chat
user_query = st.chat_input("Stelle eine Frage zu deiner Datenbank...")

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

# Hinweis zur Ausführung
st.sidebar.markdown("---")
st.sidebar.caption("Um die App zu starten: `streamlit run frontend.py` im Terminal ausführen.")
st.sidebar.caption("Stellen Sie sicher, dass `backend.py` und Ihre `.env`-Datei im selben Verzeichnis sind.")
