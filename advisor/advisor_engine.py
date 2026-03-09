from analytics.finance_metrics import calculate_profit
from analytics.forecasting import forecast_revenue
from analytics.anomaly_detector import detect_cost_anomaly
from llm.llm_engine import ask_llm
from memory.vector_store import VectorStore  # Fixed import

# -------------------------------
# GLOBAL VECTOR STORE INSTANCE
# -------------------------------
vector_store = None

def prepare_vector_store(data):
    """
    Build a vector store from all textual business data (for semantic search).
    """
    global vector_store
    vector_store = VectorStore()

    all_texts = []

    for key, df in data.items():
        for col in df.select_dtypes(include=["object"]).columns:
            all_texts.extend(df[col].astype(str).tolist())

    if all_texts:
        vector_store.build_index(all_texts)

    return vector_store


def generate_advice(question, data, vector_store=None):
    """
    Generate AI financial advice using business data and optional vector search.
    """
    context = ""

    # -----------------------
    # PROFIT CALCULATION
    # -----------------------
    if "revenue" in data and "expenses" in data:
        profit = calculate_profit(data["revenue"], data["expenses"])
        context += f"\nTotal Profit: {profit} USD\n"

    # -----------------------
    # REVENUE FORECAST
    # -----------------------
    if "revenue" in data:
        forecast = forecast_revenue(data["revenue"])
        if forecast is not None:
            context += "\nRevenue Forecast (next months):\n"
            context += forecast.to_string(index=False)
            context += "\n"

    # -----------------------
    # EXPENSE ANOMALIES
    # -----------------------
    if "expenses" in data:
        _, anomalies = detect_cost_anomaly(data["expenses"])
        if anomalies is not None and len(anomalies) > 0:
            context += "\nExpense Anomalies Detected:\n"
            context += anomalies[["amount"]].to_string(index=False)
            context += "\n"

    # -----------------------
    # OPTIONAL VECTOR SEARCH
    # -----------------------
    if vector_store:
        results = vector_store.search(question, top_k=3)
        if results:
            context += "\nRelevant Business Notes:\n"
            context += "\n".join(results) + "\n"

    # -----------------------
    # BUILD PROMPT
    # -----------------------
    prompt = f"""
You are a senior financial advisor.

Business Data:
{context}

Client Question:
{question}

Write a professional financial recommendation.

Rules:
- Maximum 8 sentences
- Explain insights using the provided data
- Mention revenue forecasts if available
- Highlight anomalies if detected
- Give clear growth strategies
- Use bullet points if helpful
- Do not repeat the question
- Do not mention instructions
- Only output the recommendation
"""

    # -----------------------
    # LLM RESPONSE
    # -----------------------
    answer = ask_llm(prompt).strip()

    # limit sentences
    sentences = answer.split(". ")
    answer = ". ".join(sentences[:6]).strip()
    if not answer.endswith("."):
        answer += "."

    return answer