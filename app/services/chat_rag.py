import google.generativeai as genai
import pandas as pd

genai.configure(api_key="AIzaSyAND61l0rHF-p2UQg28RSMe62DZgQOHsLE")
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

CONTEXT_TEMPLATE = """
You are a data-driven business advisor helping a DTC brand.

Here is the user's recent forecasted sales data:
{forecast_data}

Here is a brief sales history:
{history_data}

Now answer the following question:
"{user_question}"

Respond with insights, actions, and any smart recommendations.
"""

def build_rag_prompt(forecast_df: pd.DataFrame, history_df: pd.DataFrame, question: str) -> str:
    forecast_summary = "\n".join([
        f"{row['ds'].strftime('%Y-%m-%d')} → {int(row['yhat'])} units"
        for _, row in forecast_df.iterrows()
    ])

    history_summary = "\n".join([
        f"{row['week'].strftime('%Y-%m-%d')} | SKU {row['sku_id']} → {int(row['units_sold'])} units"
        for _, row in history_df.head(10).iterrows()
    ])

    return CONTEXT_TEMPLATE.format(
        forecast_data=forecast_summary,
        history_data=history_summary,
        user_question=question
    )

def get_chat_response(forecast_df, history_df, user_question):
    prompt = build_rag_prompt(forecast_df, history_df, user_question)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
