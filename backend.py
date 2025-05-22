# backend.py

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

def init_database() -> SQLDatabase:
  db_uri = f"sqlite:///./AngeBot.db"
  return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db: SQLDatabase, model_name: str):
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
  prompt = ChatPromptTemplate.from_template(template)
  llm = ChatOpenAI(model=model_name)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(
        schema=get_schema,
        user_city=lambda x: x['user_info'].get('city', 'Unknown'),
        user_transport_mode=lambda x: ", ".join(x['user_info'].get('transport', [])) if isinstance(x['user_info'].get('transport', []), list) else x['user_info'].get('transport', 'Unknown'),
        user_goal=lambda x: x['user_info'].get('preferences', 'Unknown')
    )
    | prompt
    | llm
    | StrOutputParser()
  )

# Geänderte get_response Funktion
def get_response(user_query: str, db: SQLDatabase, chat_history: list, model_name: str, user_info_dict: dict): # Parameter umbenannt für Klarheit
  sql_chain = get_sql_chain(db, model_name)
  
  # Aktualisierter Template-String, der flache Schlüssel erwartet
  # Annahme: Die Fehlermeldung "Expected: ... 'users[budget]'" bedeutet, dass
  # Ihr Template tatsächlich Platzhalter wie {users_budget} (oder ähnlich flach) benötigt.
  # Die `user_info` aus der SQL-Chain-Vorlage ist für den SQL-Prompt, nicht unbedingt für diesen finalen Antwort-Prompt.
  template_final_response = """
    You are a friendly and helpful AI assistant for cooking and grocery shopping, communicating in German with a Swabian accent.
    Your goal is to provide comprehensive feedback to the user about their recipe inquiry. This includes:
    1.  Listing the required ingredients.
    2.  Identifying where these ingredients can be purchased most affordably, by comparing supermarkets in the user's vicinity (user city: {users_city}).
    3.  Suggesting the "best way" to get to one or more supermarkets, considering the user's stated transport preferences ({users_transport}) and other parameters (e.g., budget from {users_budget}, desire for organic products from {users_preferences}, accessibility needs if mentioned).

    You have been provided with the user's question, the SQL query you generated, and the results from the database. You also have access to the user's profile information.

    Database Schema (for context, you don't write SQL here):
    <SCHEMA>{schema}</SCHEMA>

    User's Profile:
    Name: {users_name}
    City: {users_city}
    Transport Options: {users_transport}
    Preferences / Allergies: {users_preferences}
    Budget: {users_budget} €
    Full User Info (for AI context): {users_as_string}


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
    "Hallo {users_name}!
    Für dein Rezept '[Rezeptname]' habe ich folgende Informationen gefunden:

    **Zutaten und Preise:**
    * [Zutat 1]:
        * Am günstigsten bei [Supermarkt A] ([Adresse A]) für [Preis] €.
        * Auch erhältlich bei [Supermarkt B] ([Adresse B]) für [Preis] €.
    * [Zutat 2]:
        * Am günstigsten bei [Supermarkt C] ([Adresse C]) für [Preis] €.

    **Meine Empfehlung für dich:**
    Da du z.B. '{users_transport}' nutzt, empfehle ich dir:
    * Option 1: Wenn du alles in einem Laden kaufen möchtest, wäre [Supermarkt X] eine gute Wahl, dort bekommst du [Zutatenliste] für insgesamt [Gesamtpreis, falls sinnvoll].
    * Option 2: Um maximal zu sparen, könntest du [Zutat 1] bei [Supermarkt A] und [Zutat 2] bei [Supermarkt C] kaufen. [Supermarkt A] in der [Straße A] ist [Hinweis zur Erreichbarkeit]. [Supermarkt C] erreichst du gut [Hinweis basierend auf TransportMode].

    Beachte auch dein Ziel '{users_preferences}': Die genannten Produkte von [Supermarkt A] sind als Bio gekennzeichnet. Dein Budget liegt bei {users_budget}€.

    Lass mich wissen, wenn du weitere Fragen hast!"

    Formulate the response now:
    """
  
  prompt_final_response = ChatPromptTemplate.from_template(template_final_response)
  llm = ChatOpenAI(model=model_name)
  
  chain = (
    RunnablePassthrough.assign(
        # Die SQL-Kette erwartet 'user_info' im Eingabe-Dictionary.
        # Das Eingabe-Dictionary für chain.invoke() enthält 'user_info': user_info_dict.
        schema=lambda _: db.get_table_info(),
        query=sql_chain, # sql_chain wird mit dem gesamten Eingabe-Dict von invoke aufgerufen
                         # und greift intern auf x['user_info'] zu, wie in get_sql_chain definiert.
        # Um die Fehlermeldung "Received: ['users']" zu berücksichtigen,
        # stellen wir hier sicher, dass das Benutzer-Dictionary unter dem Schlüssel 'users' für den nächsten Schritt verfügbar ist.
        users=lambda x: x['user_info'] # x['user_info'] ist hier user_info_dict
    ).assign(
      # x enthält nun 'users' (das user_info_dict), 'schema', 'query', und die ursprünglichen invoke-Schlüssel
      response=lambda x: db.run(x["query"]),
      # Flache Schlüssel aus dem 'users'-Dictionary für das Template extrahieren:
      users_name=lambda x: x['users'].get('name', 'N/A'),
      users_city=lambda x: x['users'].get('city', 'N/A'),
      users_transport=lambda x: ", ".join(x['users'].get('transport', [])) if isinstance(x['users'].get('transport', []), list) else str(x['users'].get('transport', 'N/A')),
      users_preferences=lambda x: x['users'].get('preferences', 'N/A'),
      users_budget=lambda x: str(x['users'].get('budget', '0.0')), # Sicherstellen, dass Budget ein String ist, falls es numerisch ist
      users_as_string=lambda x: str(x['users']) # Das gesamte 'users'-Dictionary als String
    )
    | prompt_final_response
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
    "user_info": user_info_dict, # Dieser Schlüssel 'user_info' wird von der sql_chain und dem ersten .assign() oben (users=lambda x: x['user_info']) erwartet
    "model_name": model_name
  })

# Beispielhafte Nutzung (optional, für direktes Testen des Backends)
if __name__ == "__main__":
    print("Backend-Modul wird direkt ausgeführt (Testmodus).")
    load_dotenv()
    try:
        db_instance = init_database()
        print("Datenbank initialisiert.")
        
        test_chat_history = [
            AIMessage(content="Hallo! Ich bin ein SQL-Assistent. Frag mich etwas über deine Datenbank."),
            HumanMessage(content="Welche Künstler gibt es?")
        ]
        test_user_query = "Ich brauche Zutaten für Spaghetti Carbonara. Wo finde ich das am günstigsten?"
        print(f"\nTestfrage: {test_user_query}")

        test_user_info = {
            "name": "Max Mustermann",
            "city": "Berlin",
            "preferences": "Bio",
            "transport": ["Auto", "Fahrrad"],
            "age": 30,
            "budget": 50.0
        }
        
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("FEHLER: OPENAI_API_KEY nicht gefunden. Bitte in .env Datei setzen.")
        else:
            try:
                response = get_response(test_user_query, db_instance, test_chat_history, "gpt-3.5-turbo", test_user_info)
                print("\nAntwort vom Backend:")
                print(response)
            except Exception as e:
                print(f"Fehler beim Abrufen der Antwort vom LLM: {e}")
                print("Stellen Sie sicher, dass Ihr OpenAI API Key korrekt ist und Guthaben vorhanden ist.")

    except Exception as e:
        print(f"Fehler beim Initialisieren der Datenbank oder Ausführen der Tests: {e}")
        print("Stellen Sie sicher, dass die Datenbankdatei (z.B. AngeBot.db) vorhanden ist.")