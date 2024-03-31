mapper_output = {
    '1': ['python', 'used', 'high-level', 'language', 'widely', 'programming'],
    '2': ['open-source', 'analysis', 'data', 'manipulation', 'pandas', 'tool'],
    '3': ['subset', 'learning', 'intelligence', 'artificial', 'machine'],
    '4': ['micro', 'web', 'flask', 'python', 'framework', 'written']
}

def reducer():
    word_to_id = {}
    unique_id = 1

    for article_id, words in mapper_output.items():
        for word in words:
            if word not in word_to_id:
                word_to_id[word] = unique_id
                unique_id += 1

    for word, word_id in word_to_id.items():
        print(word_id, word)

if __name__ == "__main__":
    reducer()
  

#TASK2
#REDUCER
from collections import defaultdict

def reducer(section_text):
    # Initialize a defaultdict to store term frequencies for each document
    term_freq_matrix = defaultdict(dict)

    # Collect term frequencies for each document
    for doc_id, text in section_text:
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        words = text.split()
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        term_freq = defaultdict(int)
        for word in filtered_words:
            term_freq[word] += 1
        term_freq_matrix[doc_id] = term_freq

    # Print the term frequency matrix in the specified format
    output = []
    for doc_id, term_freq in term_freq_matrix.items():
        output.extend([(doc_id, freq) for freq in term_freq.values()])

    print(", ".join([f"({doc_id}, {term_freq})" for doc_id, term_freq in output]))

if __name__ == "__main__":
    section_text = [
        ("1", "Python is a widely used high-level programming language."),
        ("2", "Pandas is an open-source data analysis and manipulation tool."),
        ("3", "Machine learning is a subset of artificial intelligence."),
        ("4", "Flask is a micro web framework written in Python.")
    ]
    reducer(section_text)


#TASK2
#REDUCER
import math

# Example term frequency matrix
term_frequency_matrix = {
    1: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
    2: [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)],
    3: [(3, 1), (3, 1), (3, 1), (3, 1), (3, 1)],
    4: [(4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)]
}

# Count total number of documents
total_documents = len(term_frequency_matrix)

# Calculate document frequency
document_frequency = {}
for doc_id, term_freq_list in term_frequency_matrix.items():
    unique_terms = set(term[0] for term in term_freq_list)
    for term in unique_terms:
        document_frequency[term] = document_frequency.get(term, 0) + 1

# Calculate IDF values
idf_matrix = {}
for doc_id, term_freq_list in term_frequency_matrix.items():
    idf_list = []
    for term, freq in term_freq_list:
        idf = round(math.log(total_documents / document_frequency[term]), 2)
        idf_list.append((term, idf))
    idf_matrix[doc_id] = idf_list

# Print IDF matrix
print("IDF Matrix:")
for doc_id, idf_list in idf_matrix.items():
    print(f"Document {doc_id}:")
    for term, idf in idf_list:
        print(f"    Term: {term}, IDF: {idf}")
   


import math
import string
import nltk
import sys
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

vocabulary = {}
total_documents = 0
document_frequency = {}

for line in sys.stdin:
    doc_id, text = line.strip().split("\t")
    total_documents += 1
    unique_words = emit_unique_words(text)
    for word in unique_words:
        if word not in vocabulary:
            vocabulary[word] = [0, 0] 
        vocabulary[word][0] += 1
        document_frequency[word] = document_frequency.get(word, 0) + 1

for word, (count, _) in vocabulary.items():
    idf = round(math.log(total_documents / document_frequency[word]), 2)
    vocabulary[word][1] = idf

max_word_length = max(len(word) for word in vocabulary.keys())

print("Vocabulary:")
print("Words".ljust(15), "Count", "IDF-Value", sep="\t")
for word in sorted(vocabulary.keys()):
    count, idf = vocabulary[word]
    print(f"{word.ljust(15)}{count}\t{idf}")



#TASK4
#REDUCER
def reducer(intermediate_data):
    tf_idf_matrix = {}
    
    for word, doc_tf_idf_list in intermediate_data.items():
        tf_idf_vector = [(doc_id, tf_idf) for doc_id, tf_idf in sorted(doc_tf_idf_list)]
        tf_idf_matrix[word] = tf_idf_vector
    
    return tf_idf_matrix

if __name__ == "__main__":
    # Example intermediate data
    intermediate_data = {
        'python': [('1', 0.58), ('4', 0.29)],
        'widely': [('1', 0.39)],
        'used': [('1', 0.39)],
        'high-level': [('1', 0.39)],
        'programming': [('1', 0.39)],
        'language': [('1', 0.39)],
        'pandas': [('2', 0.39)],
        'open-source': [('2', 0.39)],
        'data': [('2', 0.39)],
        'analysis': [('2', 0.39)],
        'manipulation': [('2', 0.39)],
        'tool': [('2', 0.39)],
        'machine': [('3', 0.39)],
        'learning': [('3', 0.39)],
        'subset': [('3', 0.39)],
        'artificial': [('3', 0.39)],
        'intelligence': [('3', 0.39)],
        'flask': [('4', 0.39)],
        'micro': [('4', 0.39)],
        'web': [('4', 0.39)],
        'framework': [('4', 0.39)],
        'written': [('4', 0.39)]
    }

    # Call the reducer function
    tf_idf_matrix = reducer(intermediate_data)

    # Print the TF/IDF matrix
    print("TF/IDF matrix:")
    for word, tf_idf_vector in tf_idf_matrix.items():
        print(f"{word}: {tf_idf_vector}")




import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def make_query_vector(query_text, vocabulary):
    def preprocess_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]  # Ensure lowercase comparison for stopwords
        return filtered_words

    query_words = preprocess_text(query_text)
    query_vector = {}
    total_words = len(query_words)
    word_freq = {}
    
    for word in query_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    for word, freq in word_freq.items():
        if word in vocabulary:
            idf = vocabulary[word][1]
            tf_idf = round((freq / total_words) * idf, 2)
            query_vector[word] = tf_idf
    
    return query_vector

# Example vocabulary
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
    'opensource': [1, 1.39],
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

# Example query text
query_text = "Python is a widely used high-level programming language."

# Example usage
query_vector = make_query_vector(query_text, vocabulary)

print("Query vector:")
for word, tf_idf in query_vector.items():
    print(f"{word}: {tf_idf}")



#RELEVANCE ANALIZATOR
#REDUCER
import math
from itertools import combinations
import sys

def process_word(word, doc_data):
    doc_pairs = [(doc_id, doc_data[doc_id]) for doc_id in sorted(doc_data.keys())]
    for (doc1, tf_idf1), (doc2, tf_idf2) in combinations(doc_pairs, 2):
        similarity = calculate_cosine_similarity(tf_idf1, tf_idf2)
        print(f"{doc1}-{doc2}\t{similarity}")

def calculate_cosine_similarity(vec1, vec2):
    dot_product = sum(val1 * val2 for val1, val2 in zip(vec1.values(), vec2.values()))
    magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def reducer():
    current_word = None
    doc_data = {}

    for line in sys.stdin:
        word, doc_id_tf_idf = line.strip().split("\t")
        doc_id, tf_idf = doc_id_tf_idf.strip('()').split(',')
        doc_id = doc_id.strip()
        tf_idf = float(tf_idf.strip())

        if current_word != word:
            if current_word:
                process_word(current_word, doc_data)
            current_word = word
            doc_data = {}

        doc_data[doc_id] = tf_idf

    if current_word:
        process_word(current_word, doc_data)

if __name__ == "__main__":
    reducer()





import math
from itertools import combinations
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Function to preprocess text and emit unique words
def emit_unique_words(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    unique_words = set(filtered_words)
    return unique_words

# Function to calculate TF-IDF vectors for documents
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

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vec1, vec2):
    dot_product = sum(val1 * val2 for val1, val2 in zip(vec1.values(), vec2.values()))
    magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# Function to rank documents based on cosine similarity with the query vector
def ranker_engine(query_vector, document_vectors):
    relevance_scores = {}
    for doc_id, doc_vector in document_vectors.items():
        similarity = calculate_cosine_similarity(query_vector, doc_vector)
        relevance_scores[doc_id] = similarity
    sorted_ids = sorted(relevance_scores, key=relevance_scores.get, reverse=True)
    return sorted_ids

# Function to extract relevant content from the text corpus
def content_extractor(sorted_ids, text_corpus):
    relevant_content = []
    for doc_id in sorted_ids:
        relevant_content.append(text_corpus.get(doc_id, "Document not found."))
    return relevant_content

# Main function to perform ranking and content extraction
def main():
    # Example section text and vocabulary (use your actual data here)
    section_text = {
        "1": "Python is a widely used high-level programming language.",
        "2": "Pandas is an open-source data analysis and manipulation tool.",
        "3": "Machine learning is a subset of artificial intelligence.",
        "4": "Flask is a micro web framework written in Python."
    }
    vocabulary = {
        'analysis': 1.39,
        'artificial': 1.39,
        'data': 1.39,
        'flask': 1.39,
        'framework': 1.39,
        'high-level': 1.39,
        'intelligence': 1.39,
        'language': 1.39,
        'learning': 1.39,
        'machine': 1.39,
        'manipulation': 1.39,
        'micro': 1.39,
        'open-source': 1.39,
        'pandas': 1.39,
        'programming': 1.39,
        'python': 0.69,
        'subset': 1.39,
        'tool': 1.39,
        'used': 1.39,
        'web': 1.39,
        'widely': 1.39,
        'written': 1.39
    }

    # Example query text
    query_text = "Python is a widely used high-level programming language."
    query_vector = calculate_tf_idf(query_text, vocabulary)

    # Calculate TF-IDF vectors for documents
    document_vectors = {}
    for doc_id, text in section_text.items():
        doc_vector = calculate_tf_idf(text, vocabulary)
        document_vectors[doc_id] = doc_vector

    # Rank documents based on relevance to the query
    sorted_ids = ranker_engine(query_vector, document_vectors)

    # Extract relevant content from the text corpus
    relevant_content = content_extractor(sorted_ids, section_text)

    print("Relevant document IDs:", sorted_ids)
    print("Relevant content:")
    for content in relevant_content:
        print(content)

if __name__ == "__main__":
    main()
