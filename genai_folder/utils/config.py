import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
    SHEET_URL = os.getenv('SHEET_URL', 'https://docs.google.com/spreadsheets/d/your_sheet_id/edit')