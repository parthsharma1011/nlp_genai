# Stop Words Removal using NLTK
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# text ="""This is a great example to demonstrate basic 
#         NLP tasks using NLTK library. and also stop words removal"""
        
# stop_words = set(stopwords.words('english')) 
# # print(f"Stop Words: {stop_words}")
# words = text.lower().split()
# # print(words)
# filtered = [word for word in words if word not in stop_words] #loop, list comprehension
# print(f"Filtered Words: {filtered}")

#Stemming vs lemmatization
# from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
# words = ["running", "runner", "ran", "easily", "fairly", "fairness", "studies", "studying", "studied"]

# porter = PorterStemmer() #moderate stemming
# print("Porter Stemmer:",[porter.stem(word) for word in words])

# #Snowball Stemmer - moderate stemming, supports multiple languages, improved version of Porter
# snowball = SnowballStemmer("english")
# print("Snowball Stemmer:",[snowball.stem(word) for word in words])

# #Lancaster Stemmer - aggressive stemming
# lancaster = LancasterStemmer()
# print("Lancaster Stemmer:",[lancaster.stem(word) for word in words])

#Lemmatization

# import spacy
# nlp = spacy.load("en_core_web_sm")
# text = """55,000 years ago, the first modern humans, or Homo sapiens, 
# arrived on the Indian subcontinent from Africa.[26][27][28] 
# The earliest known modern human remains in South Asia date to 
# about 30,000 years ago.[26] After 6500 BC, evidence for 
# domestication of food crops and animals, construction of permanent 
# structures, and storage of agricultural surplus appeared in Mehrgarh 
# and other sites in Balochistan, Pakistan.[84] These gradually developed 
# into the Indus Valley Civilisation,[85][84] the first urban culture in South
# Asia,[86] which flourished during 2500–1900 BC in Pakistan and western India.[87] 
# Centred around cities such as Mohenjo-daro, Harappa, Dholavira, and Kalibangan, and 
# relying on varied forms of subsistence, the civilisation engaged robustly in crafts 
# production and wide-ranging trade.[86]"""
# doc = nlp(text)
# for token in doc:
#     print(f"Token: {token.text}, -->: {token.lemma_}")

# # import spacy
# # spacy.cli.download("en_core_web_sm")

#Tokentization
# from nltk.tokenize import word_tokenize, sent_tokenize
# import nltk
# # nltk.download('punkt_tab') #download the punkt tokenizer models
# # text ="""Natural language processing! (NLP), How's it going? I am learning NLP."""
# # tokens = word_tokenize(text)
# # print(f"Word Tokens: {tokens}")

# text = "Hello! How are you doing? I'm learning NLP. It's amazing."
# sentences = sent_tokenize(text)
# print(f"Sentences: {sentences}")

# #Bag of Words
# from sklearn.feature_extraction.text import CountVectorizer
# #spam emails 
# corpus = [
#     "Congratulations! You've won a free lottery ticket. Click here to claim your prize.",
#     "Dear user, your account has been compromised. Please reset your password immediately.",
#     "Limited time offer! Buy one get one free on all products. Don't miss out!",
#     "Hello friend, just wanted to check in and see how you're doing.",
#     "Reminder: Your appointment is scheduled for tomorrow at 10 AM."
# ]

# labels = ["spam", "ham", "spam", "ham", "ham"]
# #Create the Bag of Words model
# vectorizer = CountVectorizer()
# bow_matrix = vectorizer.fit_transform(corpus)

# print("=="*40)
# print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
# print("=="*40)
# print("\nBag of Words")
# print(bow_matrix.toarray())
# print("=="*40)

# print(f"First email : '{corpus[0]}'")
# print(dict(zip(vectorizer.get_feature_names_out(), bow_matrix.toarray()[0])))

# #ml - class 
# features - bow_matrix
# target = labels

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = ["the cat sat on the mat",
             "the dog sat on the log",
             "cats and dogs are animals"]

#matrix creation is out goal 
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

df = pd.DataFrame(tfidf_matrix.toarray(), 
                  columns=tfidf_vectorizer.get_feature_names_out(),
                  index=['Doc1', 'Doc2', 'Doc3'])

print(df.round(3))