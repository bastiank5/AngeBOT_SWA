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

def init_database() -> SQLDatabase:
  """
  Initialisiert die Datenbankverbindung.

  Args:
    database_name: Der Name der Datenbankdatei (ohne .db Erweiterung).

  Returns:
    Ein SQLDatabase-Objekt für die Datenbankinteraktion.
  """
  # Hier erstellen wir eine Verbindungs-URL für eine SQLite-Datenbank.
  db_uri = f"sqlite:///./AngeBot.db"
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
  # Beispiele haben UserId. Neue Funktion Erstellen und im beispielen ändern

  # User's Profile Information (if available):
  #   User City: {user_city}
  #   User Transport Mode: {user_transport_mode}
  #   User Shopping Goal: {user_goal}

  template = """
    You are an AI assistant specialized in helping users with their grocery shopping for cooking recipes.
    Based on the provided database schema, the user's question, and their profile information (especially their city, transport preferences, and shopping goals like 'bio' or 'budget'), write a SQL query to find the necessary information.
    The user will typically mention a recipe or a list of ingredients. Your primary goal is to find where these ingredients can be purchased, ideally at the best price, considering supermarkets in the user's city.

    <SCHEMA>{schema}</SCHEMA>

    Guidelines for SQL Query Generation:
    1.  Prioritize finding products (`ProductName`) mentioned by the user from the `Products` table.
    2.  Compare prices (`Price`) across different supermarkets (`Supermarkets`).
    3.  Filter supermarkets to be in the user's city (`UserInfo.City` = `Supermarkets.City`).
    4.  If the user mentions preferences like "Bio" or "nachhaltig" (check `UserInfo.Goal` or the question), try to filter `Products` using `Category` or `ProductName LIKE '%Bio%'`.
    5.  If multiple ingredients are requested for a recipe, you might need to query for each or structure the query to show options per ingredient. Sometimes, finding all ingredients in one store is preferred, other times the cheapest option per ingredient across multiple stores is better. Use the context.
    6.  Assume a specific `UserID` is provided or can be inferred for accessing `UserInfo`.

    Conversation History: {chat_history}
    User's Question: {question}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Example Scenario:

    Question: Ich brauche Zutaten für Spaghetti Carbonara. Wo finde ich das am günstigsten?
    User City: Berlin
    User Transport Mode: ÖPNV
    User Shopping Goal: preisgünstig einkaufen
    SQL Query:
    SELECT p.ProductName, p.Price, s.Name AS SupermarketName, s.Address
    FROM Products p
    JOIN Supermarkets s ON p.SupermarketID = s.SupermarketID
    WHERE s.City = 'Berlin' AND p.ProductName IN ('Spaghetti', 'Eier', 'Guanciale', 'Pecorino Romano') AND p.InStock > 0
    ORDER BY p.ProductName, p.Price ASC;

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

  # User's Profile (if available):
  #   User City: {user_city}
  #   User Transport Mode: {user_transport_mode}
  #   User Shopping Goal: {user_goal}
  #   Budget: {user_budget}

  template = """
    You are a friendly and helpful AI assistant for cooking and grocery shopping, communicating in German with a Swabian accent.
    Your goal is to provide comprehensive feedback to the user about their recipe inquiry. This includes:
    1.  Listing the required ingredients.
    2.  Identifying where these ingredients can be purchased most affordably, by comparing supermarkets in the user's vicinity (city).
    3.  Suggesting the "best way" to get to one or more supermarkets, considering the user's stated transport preferences and other parameters (e.g., budget, desire for organic products, accessibility needs if mentioned).

    You have been provided with the user's question, the SQL query you generated, and the results from the database. You also have access to the user's profile information.

    Database Schema (for context, you don't write SQL here):
    <SCHEMA>{schema}</SCHEMA>

    User's Original Question: {question}

    Generated SQL Query: <SQL>{query}</SQL>
    SQL Response from Database: {response}
    Conversation History: {chat_history}

    Based on all this information, formulate a helpful and actionable response in German.

    Key aspects to include in your response:
    -   **Clarity on Ingredients and Prices:** Clearly state which ingredients were found, at what prices, and in which supermarkets.
    -   **Price Comparison:** If different supermarkets offer different prices for the same or similar items, highlight this.
    -   **Supermarket Information:** Provide the names and addresses of the recommended supermarkets.
    -   **Transport Considerations:** Tailor your advice based on the user's `TransportMode`.
        -   If `TransportMode` is 'Auto', suggest driving, mention if parking could be an issue (general knowledge, not from DB).
        -   If 'zu Fuß' or 'Fahrrad', highlight closer options if discernible, or mention it's good for local trips.
        -   If 'ÖPNV', suggest checking public transport routes to the supermarket's address.
    -   **Splitting Purchases:** If ingredients are cheapest at different stores, explain this clearly. For example: "Die Nudeln sind bei Aldi (Adresse...) am günstigsten, während das Bio-Hackfleisch bei Rewe (Adresse...) preiswerter ist. Da du mit dem Auto unterwegs bist, könntest du beide Märkte anfahren. Alternativ, wenn du Zeit sparen möchtest, hat Edeka (Adresse...) beide Artikel, aber etwas teurer."
    -   **User Goals:**
        -   If the user wants "Bio" or "nachhaltig", confirm if the found products meet this (based on `Category` or `ProductName`). If not, mention it.
        -   If the user has a `Budget`, you can comment if the total cost (if calculable) fits within it, or suggest cheaper alternatives if possible.
    -   **No "Route Planning":** Do not pretend to be a route planner. Instead of "take bus X", say "Supermarkt Y in der Musterstraße ist gut mit öffentlichen Verkehrsmitteln erreichbar." or "Mit dem Auto erreichst du Supermarkt Z in der Beispielallee in etwa X Minuten (je nach Verkehr)." (X Minuten ist eine Schätzung, keine DB-Info).
    -   **Handling Missing Information:** If some ingredients are not found, or no supermarkets match certain strict criteria, state this politely and perhaps suggest alternatives or broadening the search.
    -   **Actionable Advice:** End with a helpful summary or next step.

    Beispiel für eine gute Antwortstruktur:
    "Hallo [User-Vorname, falls bekannt]!
    Für dein Rezept '[Rezeptname]' habe ich folgende Informationen gefunden:

    **Zutaten und Preise:**
    * [Zutat 1]:
        * Am günstigsten bei [Supermarkt A] ([Adresse A]) für [Preis] €.
        * Auch erhältlich bei [Supermarkt B] ([Adresse B]) für [Preis] €.
    * [Zutat 2]:
        * Am günstigsten bei [Supermarkt C] ([Adresse C]) für [Preis] €.

    **Meine Empfehlung für dich:**
    Da du [TransportMode, z.B. 'mit dem Fahrrad unterwegs bist und nicht weit fahren möchtest / ein Auto hast'], empfehle ich dir:
    * Option 1: Wenn du alles in einem Laden kaufen möchtest, wäre [Supermarkt X] eine gute Wahl, dort bekommst du [Zutatenliste] für insgesamt [Gesamtpreis, falls sinnvoll].
    * Option 2: Um maximal zu sparen, könntest du [Zutat 1] bei [Supermarkt A] und [Zutat 2] bei [Supermarkt C] kaufen. [Supermarkt A] in der [Straße A] ist [Hinweis zur Erreichbarkeit, z.B. 'nur 5 Minuten zu Fuß von deiner angegebenen Adresse entfernt' - falls ableitbar, sonst allgemeiner]. [Supermarkt C] erreichst du gut [Hinweis basierend auf TransportMode].

    Beachte auch dein Ziel '[User Goal, z.B. nur Bio kaufen]': Die genannten Produkte von [Supermarkt A] sind als Bio gekennzeichnet.

    Lass mich wissen, wenn du weitere Fragen hast!"

    Formulate the response now:
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

