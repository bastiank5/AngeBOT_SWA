# backend.py
# Diese Datei enthält die Logik für die Datenbankinteraktion und die Chat-Antwortgenerierung.

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage # Wird im Frontend benötigt, aber hier für get_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Lädt Umgebungsvariablen aus einer ".env"-Datei (z.B. API-Schlüssel für OpenAI)
# Es ist gut, dies hier zu tun, falls Backend-Funktionen direkt getestet werden
# oder falls das Backend als separater Service laufen würde.
load_dotenv()

def init_database(database_name: str) -> SQLDatabase:
  """
  Initialisiert die Datenbankverbindung.

  Args:
    database_name: Der Name der Datenbankdatei (ohne .db Erweiterung).

  Returns:
    Ein SQLDatabase-Objekt für die Datenbankinteraktion.
  """
  # Hier erstellen wir eine Verbindungs-URL für eine SQLite-Datenbank.
  db_uri = f"sqlite:///./{database_name}.db"
  # Wir erstellen ein SQLDatabase-Objekt, das später verwendet wird, um SQL-Queries abzuschicken.
  return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db: SQLDatabase):
  """
  Erstellt eine Langchain-Kette (Chain) zur Generierung von SQL-Abfragen.

  Args:
    db: Das SQLDatabase-Objekt.

  Returns:
    Eine Langchain-Kette, die SQL-Abfragen generiert.
  """
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
  # Stellen Sie sicher, dass der OPENAI_API_KEY in Ihrer .env-Datei oder Umgebungsvariablen gesetzt ist.
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

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  """
  Generiert eine vollständige, natürliche Antwort für den Benutzer.

  Args:
    user_query: Die Frage des Benutzers.
    db: Das SQLDatabase-Objekt.
    chat_history: Die bisherige Konversationshistorie.

  Returns:
    Eine textuelle Antwort für den Benutzer.
  """
  # Zuerst holen wir uns die SQL-Chain (also die Chain, die SQL-Queries generiert)
  sql_chain = get_sql_chain(db)
  
  # In diesem Prompt-Template geht es nicht mehr darum, eine SQL-Abfrage zu erstellen, sondern darum, 
  # eine nette, natürliche Antwort in Textform zu formulieren.
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
  # Stellen Sie sicher, dass der OPENAI_API_KEY in Ihrer .env-Datei oder Umgebungsvariablen gesetzt ist.
  llm = ChatOpenAI(model="gpt-4o")
  
  # Chain wird hier zusammengebaut:
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars_dict: db.run(vars_dict["query"]), # vars_dict statt vars, um Konflikte zu vermeiden
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

# Beispielhafte Nutzung (optional, für direktes Testen des Backends)
if __name__ == "__main__":
    # Dieser Code wird nur ausgeführt, wenn backend.py direkt gestartet wird.
    # Er dient zum Testen der Backend-Funktionen.
    print("Backend-Modul wird direkt ausgeführt (Testmodus).")
    
    # Laden der Umgebungsvariablen (wichtig für den API Key)
    load_dotenv()

    try:
        db_instance = init_database("AngeBot")
        print("Datenbank initialisiert.")
        
        # Test-Schema abrufen
        # print("Datenbankschema:", db_instance.get_table_info())

        # Test-Chat-Historie
        test_chat_history = [
            AIMessage(content="Hallo! Ich bin ein SQL-Assistent. Frag mich etwas über deine Datenbank."),
            HumanMessage(content="Welche Künstler gibt es?")
        ]
        
        # Test-Abfrage
        test_user_query = "Nenne mir 5 Künstler."
        print(f"\nTestfrage: {test_user_query}")
        
        # Antwort generieren
        # Überprüfen, ob der OPENAI_API_KEY geladen wurde
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("FEHLER: OPENAI_API_KEY nicht gefunden. Bitte in .env Datei setzen.")
        else:
            try:
                response = get_response(test_user_query, db_instance, test_chat_history)
                print("\nAntwort vom Backend:")
                print(response)
            except Exception as e:
                print(f"Fehler beim Abrufen der Antwort vom LLM: {e}")
                print("Stellen Sie sicher, dass Ihr OpenAI API Key korrekt ist und Guthaben vorhanden ist.")

    except Exception as e:
        print(f"Fehler beim Initialisieren der Datenbank oder Ausführen der Tests: {e}")
        print("Stellen Sie sicher, dass die Datenbankdatei (z.B. AngeBot.db) vorhanden ist.")

