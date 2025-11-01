import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import warnings
from transformers import pipeline
warnings.filterwarnings("ignore")

class FeedbackAnalyzer:
    def __init__(self):
        print(f"Loading Models...(this may take a while)")
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    