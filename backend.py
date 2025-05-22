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


def get_sql_chain(db: SQLDatabase, model_name: str):
  """
  Erstellt eine Langchain-Kette (Chain) zur Generierung von SQL-Abfragen.

  Args:
    db: Das SQLDatabase-Objekt.
    model_name: Das zu verwendende KI-Modell.

  Returns:
    Eine Langchain-Kette, die SQL-Abfragen generiert.
  """
  # Hier definieren wir eine Vorlage {template} für den Chatbot:
  # Der Chatbot soll auf Basis der existierenden Datenbankstruktur {schema} SQL-Abfragen generieren.
  # user_info wird als Dictionary erwartet.
  # template = """
  #   You are an AI assistant specialized in helping users with their grocery shopping for cooking recipes.
  #   Based on the provided database schema, the user's question, and their profile information (user_info: {user_info}), write a SQL query to find the necessary information.
  #   The user will typically mention a recipe or a list of ingredients. Your primary goal is to find where these ingredients can be purchased, ideally at the best price, considering supermarkets in the user's city (user_info['city']).
  #   If user_info['preferences'] contains 'Bio' or 'organic', prioritize organic products.
  #   If user_info['budget'] is relevant, consider it.

  #   <SCHEMA>{schema}</SCHEMA>

  #   Conversation History: {chat_history}
  #   User Info: {user_info}
  #   User's Question: {question}

  #   Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
  #   """
  template = """
    You are an AI assistant specialized in helping users with their grocery shopping for cooking recipes.
    Based on the provided database schema, the user's question, and their profile information (especially their city: {user_city}, transport preferences: {user_transport_mode}, and shopping goals like '{user_goal}'), write a SQL query to find the necessary information.
    The user will typically mention a recipe or a list of ingredients. Your primary goal is to find where these ingredients can be purchased, ideally at the best price, considering supermarkets in the user's city.

    <SCHEMA>{schema}</SCHEMA>

    Guidelines for SQL Query Generation:
    1.  Prioritize finding products (`ProductName`) mentioned by the user from the `Products` table.
    2.  Compare prices (`Price`) across different supermarkets (`Supermarkets`).
    3.  Filter supermarkets to be in the user's city. For example, if user_city is 'Berlin', use `WHERE Supermarkets.City = 'Berlin'`.
    4.  If the user mentions preferences like "Bio" or "nachhaltig" (check user_goal or the question), try to filter `Products` using `Category` or `ProductName LIKE '%Bio%'`.
    5.  If multiple ingredients are requested for a recipe, you might need to query for each or structure the query to show options per ingredient. Sometimes, finding all ingredients in one store is preferred, other times the cheapest option per ingredient across multiple stores is better. Use the context.
    6.  Assume a specific `UserID` is provided or can be inferred for accessing `UserInfo` (though `UserInfo` table itself is not directly used for filtering products here, but general user context).

    Conversation History: {chat_history}
    User's Question: {question}
    User Info: {user_info} # Full user_info dictionary for context

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
  
  # Wir verwenden das übergebene KI-Modell.
  # Stellen Sie sicher, dass der OPENAI_API_KEY in Ihrer .env-Datei oder Umgebungsvariablen gesetzt ist.
  llm = ChatOpenAI(model=model_name)
  
  # Diese kleine Funktion holt das Schema der aktuellen Datenbank.
  def get_schema(_):
    return db.get_table_info()
  
  # Die Chain wird zusammengebaut:
  # RunnablePassthrough.assign nimmt die Eingabe der Kette (ein Dictionary)
  # und fügt neue Schlüssel hinzu oder überschreibt vorhandene.
  # Die Lambdas erhalten das Eingabe-Dictionary x.
  return (
    RunnablePassthrough.assign(
        schema=get_schema, # Fügt den Datenbankschema hinzu
        # Extrahiert spezifische User-Infos für den Prompt; user_info selbst ist schon im Input-Dict
        user_city=lambda x: x['user_info'].get('city', 'Unknown'),
        user_transport_mode=lambda x: ", ".join(x['user_info'].get('transport', [])) if isinstance(x['user_info'].get('transport', []), list) else x['user_info'].get('transport', 'Unknown'),
        user_goal=lambda x: x['user_info'].get('preferences', 'Unknown') # Annahme: 'preferences' mappt zu 'goal'
    )
    | prompt # Erhält das Dictionary mit schema, user_city, user_transport_mode, user_goal und den ursprünglichen Eingaben
    | llm
    | StrOutputParser()
  )

def get_response(user_query: str, db: SQLDatabase, chat_history: list, model_name: str, user_info: dict):
  """
  Generiert eine vollständige, natürliche Antwort für den Benutzer.

  Args:
    user_query: Die Frage des Benutzers.
    db: Das SQLDatabase-Objekt.
    chat_history: Die bisherige Konversationshistorie.
    model_name: Das zu verwendende KI-Modell (z.B. "gpt-3.5-turbo" oder "gpt-4").
    user_info: Ein Dictionary mit Benutzerinformationen.

  Returns:
    Eine textuelle Antwort für den Benutzer.
  """
  # Zuerst holen wir uns die SQL-Chain (also die Chain, die SQL-Queries generiert)
  # model_name und user_info werden hier benötigt, damit sql_chain sie verwenden kann
  sql_chain = get_sql_chain(db, model_name) # user_info wird implizit durch den invoke Aufruf weitergegeben
  
  # In diesem Prompt-Template geht es nicht mehr darum, eine SQL-Abfrage zu erstellen, sondern darum, 
  # eine nette, natürliche Antwort in Textform zu formulieren.

  template = """
    You are a friendly and helpful AI assistant for cooking and grocery shopping, communicating in German with a Swabian accent.
    Your goal is to provide comprehensive feedback to the user about their recipe inquiry. This includes:
    1.  Listing the required ingredients.
    2.  Identifying where these ingredients can be purchased most affordably, by comparing supermarkets in the user's vicinity (user_info['city']).
    3.  Suggesting the "best way" to get to one or more supermarkets, considering the user's stated transport preferences (user_info['transport']) and other parameters (e.g., budget from user_info['budget'], desire for organic products from user_info['preferences'], accessibility needs if mentioned).

    You have been provided with the user's question, the SQL query you generated, and the results from the database. You also have access to the user's profile information.

    Database Schema (for context, you don't write SQL here):
    <SCHEMA>{schema}</SCHEMA>

    User's Profile:
    Name: {user_info[name]}
    City: {user_info[city]}
    Transport Options: {user_info[transport]}
    Preferences / Allergies: {user_info[preferences]}
    Budget: {user_info[budget]} €
    Full User Info (for AI context): {user_info}


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
    "Hallo {user_info[name]}!
    Für dein Rezept '[Rezeptname]' habe ich folgende Informationen gefunden:

    **Zutaten und Preise:**
    * [Zutat 1]:
        * Am günstigsten bei [Supermarkt A] ([Adresse A]) für [Preis] €.
        * Auch erhältlich bei [Supermarkt B] ([Adresse B]) für [Preis] €.
    * [Zutat 2]:
        * Am günstigsten bei [Supermarkt C] ([Adresse C]) für [Preis] €.

    **Meine Empfehlung für dich:**
    Da du z.B. '{user_info[transport]}' nutzt, empfehle ich dir:
    * Option 1: Wenn du alles in einem Laden kaufen möchtest, wäre [Supermarkt X] eine gute Wahl, dort bekommst du [Zutatenliste] für insgesamt [Gesamtpreis, falls sinnvoll].
    * Option 2: Um maximal zu sparen, könntest du [Zutat 1] bei [Supermarkt A] und [Zutat 2] bei [Supermarkt C] kaufen. [Supermarkt A] in der [Straße A] ist [Hinweis zur Erreichbarkeit]. [Supermarkt C] erreichst du gut [Hinweis basierend auf TransportMode].

    Beachte auch dein Ziel '{user_info[preferences]}': Die genannten Produkte von [Supermarkt A] sind als Bio gekennzeichnet. Dein Budget liegt bei {user_info[budget]}€.

    Lass mich wissen, wenn du weitere Fragen hast!"

    Formulate the response now:
    """
  
  # Vorlage und Modell wieder vorbereiten
  prompt = ChatPromptTemplate.from_template(template)
  # Das übergebene KI-Modell wird verwendet.
  llm = ChatOpenAI(model=model_name)
  
  # Chain wird hier zusammengebaut:
  # Die Eingabe für diese Kette ist das Dictionary von chain.invoke().
  # sql_chain erhält dieses Dictionary ebenfalls.
  # user_info und model_name sind also im Kontext für alle Teile der Kette verfügbar,
  # die sie aus dem weitergegebenen Dictionary lesen.
  chain = (
    RunnablePassthrough.assign(
        schema=lambda _: db.get_table_info(),
        query=sql_chain, # sql_chain erhält das input dict {question, chat_history, user_info, model_name}
        user_info=lambda x: x['user_info'] # Stellt sicher, dass user_info für den finalen Prompt explizit verfügbar ist
    ).assign(
      response=lambda x: db.run(x["query"]), # x enthält hier query, schema, user_info und die ursprünglichen inputs
    )
    | prompt # prompt erhält das dict mit question, chat_history, user_info, model_name, query, schema, response
    | llm
    | StrOutputParser()
  )
  
  # Die Chain wird "angestoßen" (invoke) und liefert eine Antwort zurück.
  # Alle für die Templates benötigten Schlüssel müssen hier übergeben werden.
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
    "user_info": user_info, 
    "model_name": model_name 
  })

# Beispielhafte Nutzung (optional, für direktes Testen des Backends)
if __name__ == "__main__":
    # Dieser Code wird nur ausgeführt, wenn backend.py direkt gestartet wird.
    # Er dient zum Testen der Backend-Funktionen.
    print("Backend-Modul wird direkt ausgeführt (Testmodus).")
    
    # Laden der Umgebungsvariablen (wichtig für den API Key)
    load_dotenv()

    try:
        # db_instance = init_database("AngeBot") # init_database nimmt keinen Namen mehr an
        db_instance = init_database()
        print("Datenbank initialisiert.")
        
        # Test-Schema abrufen
        # print("Datenbankschema:", db_instance.get_table_info())

        # Test-Chat-Historie
        test_chat_history = [
            AIMessage(content="Hallo! Ich bin ein SQL-Assistent. Frag mich etwas über deine Datenbank."),
            HumanMessage(content="Welche Künstler gibt es?") # Irrelevante Historie für den Testfall
        ]
        
        # Test-Abfrage
        test_user_query = "Ich brauche Zutaten für Spaghetti Carbonara. Wo finde ich das am günstigsten?"
        print(f"\nTestfrage: {test_user_query}")

        # Beispiel user_info für den Test
        test_user_info = {
            "name": "Max Mustermann",
            "city": "Berlin",
            "preferences": "Bio", # z.B. 'Bio', 'preisgünstig', 'vegan'
            "transport": ["Auto", "Fahrrad"], # Liste von Strings
            "age": 30,
            "budget": 50.0
        }
        
        # Antwort generieren
        # Überprüfen, ob der OPENAI_API_KEY geladen wurde
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("FEHLER: OPENAI_API_KEY nicht gefunden. Bitte in .env Datei setzen.")
        else:
            try:
                # model_name hier als Beispiel "gpt-3.5-turbo" oder "gpt-4" verwenden
                response = get_response(test_user_query, db_instance, test_chat_history, "gpt-3.5-turbo", test_user_info)
                print("\nAntwort vom Backend:")
                print(response)
            except Exception as e:
                print(f"Fehler beim Abrufen der Antwort vom LLM: {e}")
                print("Stellen Sie sicher, dass Ihr OpenAI API Key korrekt ist und Guthaben vorhanden ist.")

    except Exception as e:
        print(f"Fehler beim Initialisieren der Datenbank oder Ausführen der Tests: {e}")
        print("Stellen Sie sicher, dass die Datenbankdatei (z.B. AngeBot.db) vorhanden ist.")

