# - Technischer Hinweis: Der Quellcode der App kann jederzeit mittels "streamlit run app.py" gestartet werden. 
#   Danach ist die App unter der URL "http://localhost:8501/" erreichbar. 


from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st

def init_database(database: str) -> SQLDatabase:
  # Hier erstellen wir eine Verbindungs-URL für eine SQLite-Datenbank.
  # Diese Zeile weicht vom Video ab, da wir aus Komplexitätsgründen SQLite verwenden
  db_uri = f"sqlite:///./{database}.db"
  # Wir erstellen ein SQLDatabase-Objekt, das später verwendet wird, um SQL-Queries abzuschicken.
  return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
  # Hier definieren wir eine Vorlage {template} für den Chatbot:
  # Der Chatbot soll auf Basis der existierenden Datenbankstruktur {schema} SQL-Abfragen generieren.
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
  
  # Die Vorlage wird als ChatPromptTemplate gespeichert.
  prompt = ChatPromptTemplate.from_template(template)
  
  # Wir verwenden OpenAI (GPT-4o) als KI-Modell.
  llm = ChatOpenAI(model="gpt-4o")
  
  # Diese kleine Funktion holt das Schema der aktuellen Datenbank.
  def get_schema(_):
    return db.get_table_info()
  
  # Die Chain wird zusammengebaut:
  # - Wir starten mit dem Bezug des Schemas der Datenbank über die Funktion get_schema.
  # - Dann wird daraus ein Prompt erstellt.
  # - Der Prompt wird ans Sprachmodell geschickt.
  # - Die Antwort wird als einfacher String geparst.
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

# Funktion, um eine vollständige Antwort für den Benutzer bzw. die Benutzerin zu erstellen
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  # Zuerst holen wir uns die SQL-Chain (also die Chain, die SQL-Queries generiert)
  sql_chain = get_sql_chain(db)
  
  # In diesem Prompt-Template geht es nicht mehr darum, eine SQL-Abfrage zu erstellen, sondern darum, eine nette, natürliche Antwort in Textform zu formulieren.
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response in German.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
  
  # Vorlage und Modell wieder vorbereiten
  prompt = ChatPromptTemplate.from_template(template)
  llm = ChatOpenAI(model="gpt-4o")
  
  # Chain wird hier zusammengebaut:
  # - Erst wird die SQL-Abfrage erzeugt.
  # - Dann wird die SQL-Abfrage ausgeführt und das Ergebnis geholt.
  # - Dann wird ein Text daraus formuliert.
  # Warum macht man das so kompliziert mit Ketten (|)?
  # - Modularität: Jeder Schritt ist klar getrennt.
  # - Fehler leichter finden: Man kann jeden Schritt einzeln testen.
  # - Flexibilität: Man kann Schritte leicht austauschen oder neue dazwischen bauen (z.B. "Daten prüfen", "Abfragen loggen" usw.).
  # 
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      # Ein "lambda" ist einfach eine anonyme (namenlose) Funktion. Sie wird direkt an Ort und Stelle definiert, ohne dass man eine extra def schreiben muss.
      # "lambda _" bedeutet: "Ich brauche gar keinen Eingabewert". Das _ ist nur ein Platzhalter, um zu zeigen: "Mir ist egal, was hier reinkommt – ich ignoriere es."
      schema=lambda _: db.get_table_info(),
      # lambda vars bedeutet: "Ich nehme ein Eingabewert-Paket (typischerweise ein Dictionary) namens vars." Aus diesen Eingabedaten (vars) holen wir uns die erzeugte SQL-Abfrage:
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  # Die Chain wird "angestoßen" (invoke) und liefert eine Antwort zurück.
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    

# Ab hier beginnt der Teil, der die Web-App mit Streamlit aufbaut (Frontend).

# Wenn es noch keine Chat-Historie gibt, wird sie erstellt:
# Die Historie speichert alle bisherigen Nachrichten von Mensch und KI.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Lädt Umgebungsvariablen aus einer ".env"-Datei (in unserem Fall der API-Schlüssel für OpenAI)
load_dotenv()

# Grundkonfiguration der Webseite (Titel und Icon)
st.set_page_config(page_title="Chat with SQLite", page_icon=":speech_balloon:")

# Haupttitel auf der Seite
st.title("Chat with SQLite")

# Aufbau der Seitenleiste (Sidebar)
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using SQLite. Connect to the database and start chatting.")
    
    # Eingabefeld für den Namen der Datenbank und vordefinierten Werten.
    st.text_input("Database", value="chinook", key="Database")
    
    # Button, um Verbindung zur Datenbank herzustellen
    if st.button("Connect"):
        # Initialisiert die Datenbankverbindung und speichert sie
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

# Zeigt bisherige Nachrichten im Chat-Fenster an
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Eingabefeld für neue Nachrichten im Chat
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    # Neue Benutzereingabe zur Chat-Historie hinzufügen
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        # Antwort auf Benutzereingabe holen und anzeigen
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
    
    # Antwort auch zur Chat-Historie hinzufügen
    st.session_state.chat_history.append(AIMessage(content=response))
