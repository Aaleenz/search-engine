#!/usr/bin/env python
#TASK1
#MAPPER
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def emit_unique_words(article_id, text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    unique_words = set(filtered_words)
    for word in unique_words:
        yield article_id, word

section_text = [
    ("1", "Python is a widely used high-level programming language."),
    ("2", "Pandas is an open-source data analysis and manipulation tool."),
    ("3", "Machine learning is a subset of artificial intelligence."),
    ("4", "Flask is a micro web framework written in Python.")
]

for article_id, text in section_text:
    for article_id, word in emit_unique_words(article_id, text):
        print(article_id, word)




#!/usr/bin/env python
#TASK2
#MAPPER
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

print("ID", "   Word", "   Frequency")

def emit_tf(document_id, text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    words = text.split()  
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    tf = {}
    for word in filtered_words:
        if word not in tf:
            tf[word] = 1
        else:
            tf[word] += 1
    
    for word, frequency in tf.items():
        print(f"{document_id}\t{word}\t\t{frequency}")

if __name__ == "__main__":
    section_text = [
        ("1", "Python is a widely used high-level programming language."),
        ("2", "Pandas is an open-source data analysis and manipulation tool."),
        ("3", "Machine learning is a subset of artificial intelligence."),
        ("4", "Flask is a micro web framework written in Python.")
    ]

    for doc_id, text in section_text:
        emit_tf(doc_id, text)






#TASK4
#MAPPER
from collections import defaultdict
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def emit_unique_words(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    unique_words = set(filtered_words)
    return unique_words

def mapper(section_text, vocabulary):
    intermediate_data = defaultdict(list)
    
    for doc_id, text in section_text:
        unique_words = emit_unique_words(text)
        for word in unique_words:
            if word in vocabulary:
                count, idf = vocabulary[word]
                tf = text.lower().split().count(word) / len(text.split())
                tf_idf = round(tf * idf, 2)
                intermediate_data[word].append((doc_id, tf_idf))
    
    return intermediate_data

if __name__ == "__main__":
    # Example vocabulary
    vocabulary = {
        'analysis': (1, 1.39),
        'artificial': (1, 1.39),
        'data': (1, 1.39),
        'flask': (1, 1.39),
        'framework': (1, 1.39),
        'highlevel': (1, 1.39),
        'intelligence': (1, 1.39),
        'language': (1, 1.39),
        'learning': (1, 1.39),
        'machine': (1, 1.39),
        'manipulation': (1, 1.39),
        'micro': (1, 1.39),
        'opensource': (1, 1.39),
        'pandas': (1, 1.39),
        'programming': (1, 1.39),
        'python': (2, 0.69),
        'subset': (1, 1.39),
        'tool': (1, 1.39),
        'used': (1, 1.39),
        'web': (1, 1.39),
        'widely': (1, 1.39),
        'written': (1, 1.39)
    }

    # Example section text
    section_text = [
        ("1", "Python is a widely used high-level programming language."),
        ("2", "Pandas is an open-source data analysis and manipulation tool."),
        ("3", "Machine learning is a subset of artificial intelligence."),
        ("4", "Flask is a micro web framework written in Python.")
    ]

    # Call the mapper function
    intermediate_data = mapper(section_text, vocabulary)

    # Print the intermediate data
    for word, doc_tf_idf_list in intermediate_data.items():
        print(f"{word}: {doc_tf_idf_list}")



#RELEVANCE ANALIZATOR
#MAPPER
import string
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def emit_unique_words(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    unique_words = set(filtered_words)
    return unique_words

def calculate_tf_idf(text, word_idf):
    words = text.lower().split()
    word_freq = {}
    total_words = len(words)
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    tf_idf_vector = {}
    for word, freq in word_freq.items():
        if word in word_idf:
            tf_idf = round((freq / total_words) * word_idf[word], 2)
            tf_idf_vector[word] = tf_idf
    return tf_idf_vector

def mapper(section_text, vocabulary):
    word_idf = {}  # Dictionary to store IDF values

    # Calculate IDF values for the vocabulary
    total_documents = len(section_text)
    for doc_id, text in section_text:
        unique_words = emit_unique_words(text)
        for word in unique_words:
            if word not in word_idf:
                doc_freq = sum(1 for _, doc_text in section_text if word in doc_text.lower())
                word_idf[word] = math.log(total_documents / (1 + doc_freq))

    # Emit unique words and TF-IDF vectors for documents
    for doc_id, text in section_text:
        unique_words = emit_unique_words(text)
        tf_idf_vector = calculate_tf_idf(text, word_idf)
        for word in unique_words:
            if word in tf_idf_vector:
                print(f"{word}\t({doc_id}, {tf_idf_vector[word]})")

if __name__ == "__main__":
    section_text = [
        ("1", "Python is a widely used highlevel programming language."),
        ("2", "Pandas is an open-source data analysis and manipulation tool."),
        ("3", "Machine learning is a subset of artificial intelligence."),
        ("4", "Flask is a micro web framework written in Python.")
    ]

    vocabulary = {
        'analysis': [1, 1.39],
        'artificial': [1, 1.39],
        'data': [1, 1.39],
        'flask': [1, 1.39],
        'framework': [1, 1.39],
        'highlevel': [1, 1.39],
        'intelligence': [1, 1.39],
        'language': [1, 1.39],
        'learning': [1, 1.39],
        'machine': [1, 1.39],
        'manipulation': [1, 1.39],
        'micro': [1, 1.39],
        'open-source': [1, 1.39],
        'pandas': [1, 1.39],
        'programming': [1, 1.39],
        'python': [2, 0.69],
        'subset': [1, 1.39],
        'tool': [1, 1.39],
        'used': [1, 1.39],
        'web': [1, 1.39],
        'widely': [1, 1.39],
        'written': [1, 1.39]
    }

    mapper(section_text, vocabulary)
