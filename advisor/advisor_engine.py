from analytics.finance_metrics import calculate_profit
from llm.llm_engine import ask_llm

def generate_advice(question, data):

    context = ""

    if "revenue" in data and "expenses" in data:
        profit = calculate_profit(data["revenue"], data["expenses"])
        context = f"The company profit is {profit} USD."

    prompt = f"""
You are a senior financial advisor.

Business Context:
{context}

Client Question:
{question}

Write a professional investment recommendation.

Strict Rules:
- Maximum 10 sentences
- Explain clearly with accurate values.
- Forecast the Revenue 
- Find Anomolies if exists
- Give clear advices as for growth perspectively
- If needed bullet points
- With Finance Insights Explanations 
- No introduction like "I am a financial advisor"
- Do not repeat the question
- Only output the recommendation
"""

    answer = ask_llm(prompt).strip()

    # limit output to 4 sentences
    sentences = answer.split(". ")
    answer = ". ".join(sentences[:4]).strip()

    if not answer.endswith("."):
        answer += "."

    return answer