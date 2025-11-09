# # from transformers import pipeline

# # def sentiment_analysis_demo(text):
# #     classifier = pipeline("sentiment-analysis")
# #     result = classifier(text)
# #     print(result)
# #     return result

# # sa = sentiment_analysis_demo("I love natural language processing!")
# # print(sa)

# # #instagram -> ritu i love your product but its not long lasting
# # from transformers import pipeline

# # def sentiment_analysis_demo(text):
# #     classifier = pipeline("sentiment-analysis")
# #     result = classifier(text)
# #     return result

# # sa = sentiment_analysis_demo("but this to the then than")
# # sa

# # #zero-shot classification
# # def zero_shot_classification_demo():
# #   classifier = pipeline("zero-shot-classification") #distilroberts
# #   text = "I think i will be making pizza tonight" #364
# #   candiate_lables = ["technology","war","germany",'cheese']
# #   result = classifier(text, candiate_lables)
# #   return result

# # #model shards -> model tensors
# # zsc = zero_shot_classification_demo()
# # zsc

# #NER
# import spacy
# from spacy import displacy

# # def basic_ner_demo():
# #     nlp = spacy.load("en_core_web_lg") #sm = small, md = medium, lg = large
# #     # texts = [
# #     #     "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
# #     #     "Elon Must is the CEO of SpaceX and Tesla Motors.",
# #     #     "The Eiffel Tower is located in Paris, France.",
# #     #     "Bharat spends $5 billion on defense every year. including 2025"
# #     # ]
# #     texts=["""Subject: Urgent: Follow-up on Delayed Order Placed on 22/10/2025

# # Dear [Company Name] Customer Support,

# # I hope this message finds you well. I am writing to express my concern regarding an order I placed on 22nd October 2025, which has yet to be delivered or made available for pickup. I reside in Delhi, and the delay is causing considerable inconvenience.

# # Despite receiving an initial confirmation, I have not received any updates on the status of the shipment. I understand that occasional delays may occur, but it has now been several days beyond the expected delivery window, and I would appreciate some clarity on the situation.

# # To help expedite the process, I am available for pickup on the following dates and times:

# # 27th October 2025 (Monday) – Anytime between 10:00 AM and 4:00 PM

# # 28th October 2025 (Tuesday) – After 2:00 PM
# # 30th October 2025 (Thursday) – Full day availability

# # Please let me know if any of these slots work for your team, or if there are alternative arrangements you can offer. I would also appreciate a tracking update or estimated delivery timeline if pickup is not feasible.

# # Looking forward to your prompt response and resolution.

# # Warm regards, [Your Full Name] [Your Contact Number] [Your Email Address] Delhi, India"""]
# #     for text in texts:
# #         doc = nlp(text)
        
# #         if doc.ents:
# #             print(f"Entities in '{text}':")
# #             for ent in doc.ents:
# #                 print(f"  - {ent.text} ({ent.label_}) ")
# #             print("\n")
# #         else:
# #             print(f"No entities found in '{text}'\n")

# # ner_demo = basic_ner_demo()
# # print(ner_demo)

# def detailed_ner_demo():
#     nlp = spacy.load("en_core_web_lg")
#     # text = """Google was founded in 1998, by Larry Page and 
#     # sergey Brin while they were Ph.D. students at Stanford University in California.
#     # The company employs over 150,000 people and is headquartered in Mountain View,
#     # California."""
    
#     # replaced with datase - query -> auto
#     text = """Subject: Urgent: Follow-up on Delayed Order Placed on 22/10/2025

# # Dear [Company Name] Customer Support,

# # I hope this message finds you well. I am writing to express my concern regarding an order I placed on 22nd October 2025, which has yet to be delivered or made available for pickup. I reside in Delhi, and the delay is causing considerable inconvenience.

# # Despite receiving an initial confirmation, I have not received any updates on the status of the shipment. I understand that occasional delays may occur, but it has now been several days beyond the expected delivery window, and I would appreciate some clarity on the situation.

# # To help expedite the process, I am available for pickup on the following dates and times:

# # 27th October 2025 (Monday) – Anytime between 10:00 AM and 4:00 PM
# # 28th October 2025 (Tuesday) – After 2:00 PM
# # 30th October 2025 (Thursday) – Full day availability
# 25/10/25 - 

# # Please let me know if any of these slots work for your team, or if there are alternative arrangements you can offer. I would also appreciate a tracking update or estimated delivery timeline if pickup is not feasible.

# # Looking forward to your prompt response and resolution.

# # Warm regards, [Your Full Name] [Your Contact Number] [Your Email Address] Delhi, India"""
#     doc = nlp(text)
    
#     label_explanations = {
#         "PERSON": "People, including fictional.",
#         "ORG":"Companies, agencies, institutions, etc.",
#         "GPE":"Countries, cities, states.",
#         "DATE":"Absolute or relative dates or periods.",
#         "CARDINAL":"Numerals that do not fall under another type.",
#         "MONEY":"Monetary values, including unit.",
#         "ORDINAL":"“first”, “second”, etc.",
#         "TIME":"Times smaller than a day.",
#         "PERCENT":"Percentage, including “%”.",
#         "QUANTITY":"Measurements, as of weight or distance.",
#         "LOC":"Non-GPE locations, mountain ranges, bodies of water.",
#         "FAC":"Buildings, airports, highways, bridges, etc.",
#         "PRODUCT":"Objects, vehicles, foods, etc. (Not services.)",
#         "WORK_OF_ART":"Titles of books, songs, etc.",
#         "EVENT":"Named hurricanes, battles, wars, sports events, etc.",
#         "LAW":"Named documents made into laws.",
#         "LANGUAGE":"Any named language."
#     }
    
#     for ent in doc.ents:
#         explanation = label_explanations.get(ent.label_, "")
#         print(f"{ent.text:<25} {ent.label_:<15} {explanation}")
        
# ner = detailed_ner_demo()
# print(ner)

#### Parts of Speech Tagging
# import nltk
# from nltk import pos_tag, word_tokenize
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

# text = "i hate your sevice and the product is not good enough for me"
# print("=="*40)
# tokens = word_tokenize(text)
# print(tokens)
# print("=="*40)
# tags = pos_tag(tokens)
# print("=="*40)
# print(tags)
# print("=="*40)

# #topic modelling
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation

# documents = [
#     "Python programming is great for data science and machine learning",
#     "I love playing basketball and football with my friends",
#     "Cooking pasta with tomoato sauce is delicious",
#     "machine learning algorithms need lots of data to perform well",
#     "Soccer and tennis are poprular sports worldwide",
#     "Baking cakes required flour, sugar, and eggs",
#     "Deep learning used neural networks to model complex patterns in data",
#     "Swimming and running keep you healthy and fit",
#     "French cuisine includes dishes like croissants and ratatouille",
# ]

# print("== "*40)
# print("Topic Modeling Demo")
# print("== "*40)

# print(f"\nWe have {len(documents)} documents:\n")

# #Step 1: Convert text data into a numbers
# vectorizer = CountVectorizer(stop_words='english')
# doc_matrix = vectorizer.fit_transform(documents)
# print(f"    Document-Term Matrix Shape: {doc_matrix.shape}\n")

# #Step 2: find topics - top -3 
# num_topics = 3
# lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
# lda_model.fit(doc_matrix)

#visualize topics
# words = vectorizer.get_feature_names_out()
# print("== "*40)
# print(words)
# print("== "*40)
# for topic_num, topic in enumerate(lda_model.components_):
#     top_words_indices = topic.argsort()[-5:][::-1]  # Top 5 words for each topic
#     top_words = [words[i] for i in top_words_indices]
#     print(f"Topic {topic_num + 1}: {', '.join(top_words)}")


# #classification - new document
# new_docs = ["I enjoy coding in python and building machine learning models",
#             "Baking bread and pastries is my favorite hobby",
#             "i enjoe coding in java and eat a lots of pasta"]

# for new_doc in new_docs:
#     new_matrix = vectorizer.transform([new_doc])
#     topic_probs = lda_model.transform(new_matrix)[0]
    
#     print(f"\nDocument: '{new_doc}'")
#     print(f"Topic Probabilities\n")
#     for i, prob in enumerate(topic_probs):
#         print(f"    Topic {i + 1}: {prob:.4f}")

#Car dmain - cust upload compliace data and segment them based on topics - fraud, kyc, account management
# yes blogs can be used for topic modelling 


#part of speech - using spacy
import spacy
nlp = spacy.load("en_core_web_sm")
text = "I had a flight from New York to San Francisco on 21st December 2023"
doc = nlp(text)
print("== "*40)
print(f"{'Word':<15} {'Lemma':<15} {'POS':<10} {'Tag':<10} {'Explanation'}")
for token in doc:
    explanation = spacy.explain(token.tag_) or "N/A"
    print(f"{token.text:<15} {token.lemma_:<15} {token.pos_:<10} {token.tag_:<10} {spacy.explain(token.tag_)}")
    
    
    
    
#flair
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

# make example sentence
text = """Subject: Urgent: Follow-up on Delayed Order Placed on 22/10/2025

# Dear [Company Name] Customer Support,
25/10/25 - 11 units/24 units
# I hope this message finds you well. I am writing to express my concern regarding an order I placed on 22nd October 2025, which has yet to be delivered or made available for pickup. I reside in Delhi, and the delay is causing considerable inconvenience.

# Despite receiving an initial confirmation, I have not received any updates on the status of the shipment. I understand that occasional delays may occur, but it has now been several days beyond the expected delivery window, and I would appreciate some clarity on the situation.

# To help expedite the process, I am available for pickup on the following dates and times:

# 27th October 2025 (Monday) – Anytime between 10:00 AM and 4:00 PM
# 28th October 2025 (Tuesday) – After 2:00 PM
# 30th October 2025 (Thursday) – Full day availability

# Please let me know if any of these slots work for your team, or if there are alternative arrangements you can offer. I would also appreciate a tracking update or estimated delivery timeline if pickup is not feasible.

# Looking forward to your prompt response and resolution.

# Warm regards, [Your Full Name] [Your Contact Number] [Your Email Address] Delhi, India"""
sentence = Sentence(text)

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)