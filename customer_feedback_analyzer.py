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
        self.nlp = spacy.load("en_core_web_lg")
        print(f"Models Loaded Successfully.")
        
    def analyze_sentiment(self, reviews):
        results = []
        for review in reviews:
            sentiment = self.sentiment_analyzer(review)[0] #clip idea 
            results.append({
                "review": review,
                "label": sentiment['label'],
                "confidence": round(sentiment['score'],2)
            })
        return results
    
    def extract_entities(self, reviews):
        all_entities = {
            "PRODUCT":[],
            "ORG:":[],
            "GPE:":[],
            "PERSON:":[]   
        }
        for review in reviews:
            doc = self.nlp(review)
            for ent in doc.ents:
                if ent.label_ in all_entities:
                    all_entities[ent.label_].append(ent.text)
        entity_counts = {}
        for ent_type, entities in all_entities.items():
            if entities:
                entity_counts[ent_type] = Counter(entities).most_common(5)
        return entity_counts
    
    def discover_topics(self, reviews, num_topics=3):
        vectorizer = CountVectorizer(stop_words='english')
        try:
            doc_matrix = vectorizer.fit_transform(reviews)
            lda = LatentDirichletAllocation(n_components=num_topics, 
                                            random_state=42,max_iter=10)
            lda.fit(doc_matrix)
            words = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-5:][::-1]
                top_words = [words[i] for i in top_words_idx]
                topics.append({
                    "topic_number": topic_idx + 1,
                    "keywords": top_words})
            return topics
        except Exception as e:
            print(f"Error in topic discovery: {e}")
            
    def get_summary_stats(self, sentiment_results):
        sentiments = [r['label'] for r in sentiment_results]
        total = len(sentiments)
        positive = sentiments.count('POSITIVE')
        negative = sentiments.count('NEGATIVE')
        return {
            "total_reviews": total,
            "positive_reviews": positive,
            "negative_reviews": negative,
            "positive_percentage": round(positive / total * 100, 2),
            "negative_percentage": round(negative / total * 100, 2)
        }
        
    def analyze_all(self, reviews):
        print("=="*50)
        print(f"CUSTOMER FEEDBACK ANALYSIS REPORT")
        print("=="*50)
        
        #1. Sentiment Analysis
        print("\n1. Sentiment Analysis")
        sentiment_results = self.analyze_sentiment(reviews)
        stats = self.get_summary_stats(sentiment_results)
        
        #2. Entity Extraction
        print("\n2. Entity Extraction")
        entities = self.extract_entities(reviews)
        
        #3. Topic Discovery - topic modelling
        print("\n3. Topic Discovery")
        topics = self.discover_topics(reviews)
        
        print(f"Analysis Complete!.\n")
        
        return {
            "sentiment_results": sentiment_results,
            "stats": stats,
            "entities": entities,
            "topics": topics
        }


def print_results(results):
    print("=="*50)
    print("Summary Statistics:")
    stats = results['stats']
    print(f"Total Reviews: {stats['total_reviews']}")
    print(f"Positive Reviews: {stats['positive_reviews']} ({stats['positive_percentage']}%)")
    print(f"Negative Reviews: {stats['negative_reviews']} ({stats['negative_percentage']}%)")
    
    #Sentiment Details
    print("\n" + "=="*20 + " Individual review Sentiments " + "=="*20)
    for i, result in enumerate(results['sentiment_results'][:5],1):
        sentiment_emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
        print(f"\n{i}. Review: {result['review']}\n   Sentiment: {result['label']} {sentiment_emoji} (Confidence: {result['confidence']})")
    
    if len(results['sentiment_results']) > 5:
        print(f"\n... and {len(results['sentiment_results']) - 5} more reviews analyzed.")
        
    #Topics Details
    print("\n" + "=="*20 + " Topic Discovery " + "=="*20)
    for topic in results['topics']:
        print(f"\nTopic {topic['topic_number']}: " + ", ".join(topic['keywords']))
        
    #Entities Details
    print("\n" + "=="*20 + " Extracted Entities " + "=="*20)
    entity_labels = {
        "PRODUCT":"Products",
        "ORG:":"Organizations",
        "GPE:":"Locations",
        "PERSON:":"People menationed"
    }
    entities = results['entities']
    if entities:
        for ent_type, labels in entity_labels.items():
            if ent_type in entities:
                print(f"\n{labels}:")
                for entity, count in entities[ent_type]:
                    print(f" - {entity} (mentioned {count} times)")
    else:
        print("No significant entities found.")

if __name__ == "__main__":
    sample_reviews = [
        "I absolutely love the new smartphone I bought! The camera quality is amazing and the battery life lasts all day.",
        "The customer service at the store was terrible. I waited for an hour and no one helped me.",
        "Great value for the price. The laptop performs well for my daily tasks and the build quality feels solid.",
        "I'm very disappointed with the headphones. The sound quality is poor and they are uncomfortable to wear.",
        "The delivery was quick and the packaging was secure. I'm satisfied with my purchase.",
        "This restaurant has the best pasta I've ever had! The ambiance is cozy and the staff are friendly.",
        "I had a bad experience with the airline. My flight was delayed for hours without any explanation.",
        "The new software update improved the app's performance significantly. It's much faster now.",
        "I don't recommend this vacuum cleaner. It stopped working after just two months of use.",
        "Fantastic experience at the hotel! The rooms were clean and the view was breathtaking."
    ]
    
    analyzer = FeedbackAnalyzer()
    analysis_results = analyzer.analyze_all(sample_reviews)
    print_results(analysis_results)
    