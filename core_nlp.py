# from transformers import pipeline

# def sentiment_analysis_demo(text):
#     classifier = pipeline("sentiment-analysis")
#     result = classifier(text)
#     print(result)
#     return result

# sa = sentiment_analysis_demo("I love natural language processing!")
# print(sa)

# #instagram -> ritu i love your product but its not long lasting
# from transformers import pipeline

# def sentiment_analysis_demo(text):
#     classifier = pipeline("sentiment-analysis")
#     result = classifier(text)
#     return result

# sa = sentiment_analysis_demo("but this to the then than")
# sa

# #zero-shot classification
# def zero_shot_classification_demo():
#   classifier = pipeline("zero-shot-classification") #distilroberts
#   text = "I think i will be making pizza tonight" #364
#   candiate_lables = ["technology","war","germany",'cheese']
#   result = classifier(text, candiate_lables)
#   return result

# #model shards -> model tensors
# zsc = zero_shot_classification_demo()
# zsc

#NER
import spacy
from spacy import displacy

# def basic_ner_demo():
#     nlp = spacy.load("en_core_web_lg") #sm = small, md = medium, lg = large
#     # texts = [
#     #     "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
#     #     "Elon Must is the CEO of SpaceX and Tesla Motors.",
#     #     "The Eiffel Tower is located in Paris, France.",
#     #     "Bharat spends $5 billion on defense every year. including 2025"
#     # ]
#     texts=["""Subject: Urgent: Follow-up on Delayed Order Placed on 22/10/2025

# Dear [Company Name] Customer Support,

# I hope this message finds you well. I am writing to express my concern regarding an order I placed on 22nd October 2025, which has yet to be delivered or made available for pickup. I reside in Delhi, and the delay is causing considerable inconvenience.

# Despite receiving an initial confirmation, I have not received any updates on the status of the shipment. I understand that occasional delays may occur, but it has now been several days beyond the expected delivery window, and I would appreciate some clarity on the situation.

# To help expedite the process, I am available for pickup on the following dates and times:

# 27th October 2025 (Monday) – Anytime between 10:00 AM and 4:00 PM

# 28th October 2025 (Tuesday) – After 2:00 PM
# 30th October 2025 (Thursday) – Full day availability

# Please let me know if any of these slots work for your team, or if there are alternative arrangements you can offer. I would also appreciate a tracking update or estimated delivery timeline if pickup is not feasible.

# Looking forward to your prompt response and resolution.

# Warm regards, [Your Full Name] [Your Contact Number] [Your Email Address] Delhi, India"""]
#     for text in texts:
#         doc = nlp(text)
        
#         if doc.ents:
#             print(f"Entities in '{text}':")
#             for ent in doc.ents:
#                 print(f"  - {ent.text} ({ent.label_}) ")
#             print("\n")
#         else:
#             print(f"No entities found in '{text}'\n")

# ner_demo = basic_ner_demo()
# print(ner_demo)

def detailed_ner_demo():
    nlp = spacy.load("en_core_web_lg")
    # text = """Google was founded in 1998, by Larry Page and 
    # sergey Brin while they were Ph.D. students at Stanford University in California.
    # The company employs over 150,000 people and is headquartered in Mountain View,
    # California."""
    
    # replaced with datase - query -> auto
    text = """Subject: Urgent: Follow-up on Delayed Order Placed on 22/10/2025

# Dear [Company Name] Customer Support,

# I hope this message finds you well. I am writing to express my concern regarding an order I placed on 22nd October 2025, which has yet to be delivered or made available for pickup. I reside in Delhi, and the delay is causing considerable inconvenience.

# Despite receiving an initial confirmation, I have not received any updates on the status of the shipment. I understand that occasional delays may occur, but it has now been several days beyond the expected delivery window, and I would appreciate some clarity on the situation.

# To help expedite the process, I am available for pickup on the following dates and times:

# 27th October 2025 (Monday) – Anytime between 10:00 AM and 4:00 PM
# 28th October 2025 (Tuesday) – After 2:00 PM
# 30th October 2025 (Thursday) – Full day availability
25/10/25 - 

# Please let me know if any of these slots work for your team, or if there are alternative arrangements you can offer. I would also appreciate a tracking update or estimated delivery timeline if pickup is not feasible.

# Looking forward to your prompt response and resolution.

# Warm regards, [Your Full Name] [Your Contact Number] [Your Email Address] Delhi, India"""
    doc = nlp(text)
    
    label_explanations = {
        "PERSON": "People, including fictional.",
        "ORG":"Companies, agencies, institutions, etc.",
        "GPE":"Countries, cities, states.",
        "DATE":"Absolute or relative dates or periods.",
        "CARDINAL":"Numerals that do not fall under another type.",
        "MONEY":"Monetary values, including unit.",
        "ORDINAL":"“first”, “second”, etc.",
        "TIME":"Times smaller than a day.",
        "PERCENT":"Percentage, including “%”.",
        "QUANTITY":"Measurements, as of weight or distance.",
        "LOC":"Non-GPE locations, mountain ranges, bodies of water.",
        "FAC":"Buildings, airports, highways, bridges, etc.",
        "PRODUCT":"Objects, vehicles, foods, etc. (Not services.)",
        "WORK_OF_ART":"Titles of books, songs, etc.",
        "EVENT":"Named hurricanes, battles, wars, sports events, etc.",
        "LAW":"Named documents made into laws.",
        "LANGUAGE":"Any named language."
    }
    
    for ent in doc.ents:
        explanation = label_explanations.get(ent.label_, "")
        print(f"{ent.text:<25} {ent.label_:<15} {explanation}")
        
ner = detailed_ner_demo()
print(ner)

