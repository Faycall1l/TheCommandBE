import google.generativeai as genai
import os

# Setup Gemini API
genai_key = "apikey"
genai.configure(api_key=genai_key)

model = genai.GenerativeModel('models/gemini-1.5-pro-latest')


PROMPT_TEMPLATE = """
You're a media strategist for a direct-to-consumer (DTC) brand in the {niche} niche. 
Based on the projected sales data below, that we got from an already established business, generate a media plan for the next 4 weeks, with a posting schedule, provide when to post each post. 

Plan should include:
- Social media post ideas (e.g. IG, TikTok, YouTube Shorts)
- Promo drop suggestions
- Influencer content timing
- Email subject ideas
- Ad timing (if relevant)

Sales Forecast:
{forecast_data}

Be smart. Recommend creative and timely ideas to drive conversions.
Return the plan as a bullet point list.
"""

def generate_media_plan(forecast_data: str, niche: str):
    prompt = PROMPT_TEMPLATE.format(
        forecast_data=forecast_data,
        niche=niche
    )
    try:
        response = model.generate_content(prompt)
        return response.text if response.parts else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"
