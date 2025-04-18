import google.generativeai as genai
import pandas as pd
from typing import List, Dict

# Configure AI model
genai.configure(api_key="AIzaSyAND61l0rHF-p2UQg28RSMe62DZgQOHsLE")
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

PROMPT_TEMPLATE = """
**Retail Sales Advisor Analysis**

**Historical Sales Summary (Last 10 Weeks):**
{history_summary}

**Sales Forecast (Next {forecast_weeks} Weeks):**
{forecast_summary}

**Business Question:**
"{user_question}"

**Provide recommendations considering:**
1. Recent sales trends
2. Forecasted demand
3. Inventory planning
4. Marketing opportunities
5. Potential risks

Format your response with clear sections and bullet points.
"""

def format_history_data(history_df: pd.DataFrame) -> str:
    """Format historical sales data for the prompt"""
    if history_df.empty:
        return "No historical data available"
    
    return "\n".join(
        f"- Week {row['week'].strftime('%Y-%m-%d')}: {int(row['units_sold'])} units "
        f"(Price: ${row['total_price']:.2f}, Featured: {'Yes' if row['is_featured_sku'] else 'No'})"
        for _, row in history_df.sort_values('week', ascending=False).head(10).iterrows()
    )

def format_forecast_data(forecast_data: List[Dict]) -> str:
    """Format forecast data for the prompt"""
    if not forecast_data:
        return "No forecast data available"
    
    return "\n".join(
        f"- {item['date']}: Projected {int(item['forecast'])} units "
        f"(Range: {int(item['forecast_lower'])}-{int(item['forecast_upper'])})"
        for item in forecast_data
    )

def get_chat_response(forecast_data: List[Dict], history_df: pd.DataFrame, question: str) -> str:
    """Generate business advice using sales data and forecasts"""
    try:
        prompt = PROMPT_TEMPLATE.format(
            history_summary=format_history_data(history_df),
            forecast_summary=format_forecast_data(forecast_data),
            forecast_weeks=len(forecast_data),
            user_question=question.strip()
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        error_msg = f"AI service error: {str(e)}"
        print(error_msg)
        return f"Could not generate advice. {error_msg}"