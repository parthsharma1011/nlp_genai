# NLP GenAI Project

A comprehensive Natural Language Processing (NLP) project demonstrating various NLP techniques using NLTK, spaCy, scikit-learn, and Transformers.

## Features

- **Text Preprocessing**: Stop words removal, stemming, lemmatization, tokenization
- **Feature Extraction**: Bag of Words (BoW), TF-IDF vectorization
- **Named Entity Recognition (NER)**: Extract entities from text using spaCy
- **Sentiment Analysis**: Analyze sentiment using Transformers
- **Customer Feedback Analysis**: Specialized class for analyzing customer feedback

## Files Overview

### `basics.py`
Contains fundamental NLP operations:
- Stop words removal using NLTK
- Stemming (Porter, Snowball, Lancaster)
- Lemmatization using spaCy
- Tokenization (word and sentence)
- Bag of Words implementation
- TF-IDF vectorization

### `core_nlp.py`
Advanced NLP functionality:
- Named Entity Recognition (NER) using spaCy
- Detailed entity extraction with explanations
- Support for multiple entity types (PERSON, ORG, GPE, DATE, etc.)

### `customer_feedback_analyzer.py`
Specialized class for customer feedback analysis:
- Sentiment analysis using Transformers
- Topic modeling capabilities
- Feedback classification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/parthsharma1011/nlp_genai.git
cd nlp_genai
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

4. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
```

## Usage

### Basic NLP Operations
```python
# Run basic NLP examples
python basics.py
```

### Named Entity Recognition
```python
# Run NER examples
python core_nlp.py
```

### Customer Feedback Analysis
```python
from customer_feedback_analyzer import FeedbackAnalyzer

# Initialize analyzer
analyzer = FeedbackAnalyzer()

# Analyze sentiment
text = "I love this product!"
result = analyzer.sentiment_analyzer(text)
print(result)
```

## Requirements

- Python 3.7+
- See `requirements.txt` for complete package list

## Models Used

- **spaCy**: `en_core_web_sm`, `en_core_web_lg`
- **Transformers**: Default sentiment analysis pipeline
- **scikit-learn**: CountVectorizer, TfidfVectorizer, LatentDirichletAllocation

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.