from transformers import pipeline


class LLMEngine:
    """
    Lightweight LLM engine using HuggingFace pipeline.
    Uses TinyLlama which runs well on CPU.
    """

    def __init__(self, device=-1):
        """
        device = -1 -> CPU
        device = 0  -> GPU
        """

        try:
            self.generator = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device=device
            )

        except Exception as e:
            print("LLM model failed to load:", e)
            self.generator = None

    def generate_insight(self, insights, user_query):
        """
        Generates AI financial analysis based on system insights
        and user query.
        """

        if self.generator is None:
            return "LLM model is not available."

        prompt = f"""
You are a professional AI Financial Advisor.

Business Analysis Results:
{insights}

User Question:
{user_query}

Instructions:
- Explain the financial situation clearly
- Identify potential risks
- Provide practical recommendations
- Keep the response concise and professional
- Do NOT repeat the prompt

Final Answer:
"""

        try:

            result = self.generator(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                pad_token_id=50256
            )

            generated_text = result[0]["generated_text"]

            # Remove prompt part from response
            answer = generated_text.replace(prompt, "").strip()

            return answer

        except Exception as e:

            print("LLM generation error:", e)

            return "Unable to generate AI financial advice at the moment."