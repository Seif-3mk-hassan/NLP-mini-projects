from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import numpy as np
from gensim.models import Word2Vec
import pickle
import os


BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH             = os.path.join(BASE_DIR, "data", "all_tickets_processed_improved_v3.csv")
TFIDF_MODEL_PATH      = os.path.join(BASE_DIR, "models", "tfidf_baseline_model.pkl")
WORD2VEC_MODEL_PATH   = os.path.join(BASE_DIR, "models", "word2vec_model.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(BASE_DIR, "src", "tfidf_vectorizer.pkl")
REPORT_PATH           = os.path.join(BASE_DIR, "reports", "evaluation_report_m1.txt")


'''Importing the dataset and performing basic cleaning and preprocessing'''
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\W', ' ', text)
        text = BeautifulSoup(text, "html.parser").get_text()
        return text
    
    data['Document'] = data['Document'].apply(clean_text)
    return data['Document'].tolist(), data['Topic_group'].tolist()


'''Tokenization, stop word removal, stemming, and lemmatization'''
def tokenize_and_preprocess(text, is_tfidf=True):
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    if is_tfidf:
        return ' '.join(tokens)  # TF-IDF needs string
    else:
        return tokens            # Word2Vec needs list of tokens


'''Convert document to vector using Word2Vec'''
def document_vector(doc, model):
    vectors = []
    for word in doc:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)


'''TF-IDF or Word2Vec vectorization'''
def vectorize_text(X, y, is_TFIDF=True, word2vec_model=None):
    
    if is_TFIDF:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train)
        X_test  = vectorizer.transform(X_test)
        
        # ✅ Correct path with filename
        '''with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✅ TF-IDF Vectorizer saved to: {TFIDF_VECTORIZER_PATH}")'''
        
        return X_train, X_test, y_train, y_test
    
    else:
        # ✅ Now model is properly passed as parameter
        if word2vec_model is None:
            raise ValueError("word2vec_model must be provided when is_TFIDF=False")
        
        X_vectors = np.array([
            document_vector(doc, word2vec_model) 
            for doc in X
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectors, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test


'''Train model and save'''
def train_model(X_train, y_train, save_path):
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # ✅ Correct path with filename
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {save_path}")
    
    return model


'''Model evaluation and save report'''
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    
    # ✅ Save report to file
    with open(REPORT_PATH, 'a') as f:  # 'a' = append so both reports saved
        f.write(f"\n{'='*50}\n")
        f.write(f"{model_name} Evaluation\n")
        f.write(f"{'='*50}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"✅ Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    
    # ✅ Create directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

    # Load data
    texts, labels = load_and_preprocess_data(DATA_PATH)

    # ──────────────────────────────────────────
    # Pipeline 1: TF-IDF + Logistic Regression
    # ──────────────────────────────────────────
    print("\n🔄 Running TF-IDF Pipeline...")
    
    tfidf_texts = [tokenize_and_preprocess(text, is_tfidf=True) for text in texts]
    
    X_train, X_test, y_train, y_test = vectorize_text(
        tfidf_texts, labels, 
        is_TFIDF=True
    )
    
    tfidf_model = train_model(X_train, y_train, TFIDF_MODEL_PATH)
    
    evaluate_model(
        tfidf_model, X_test, y_test,
        model_name="TF-IDF + Logistic Regression"
    )

    # ──────────────────────────────────────────
    # Pipeline 2: Word2Vec + Logistic Regression
    # ──────────────────────────────────────────
    print("\n🔄 Running Word2Vec Pipeline...")
    
    w2v_texts = [tokenize_and_preprocess(text, is_tfidf=False) for text in texts]
    
    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=w2v_texts, 
        vector_size=100, 
        window=5, 
        min_count=2, 
        workers=4, 
        epochs=10, 
        sg=1
    )
    
    # ✅ Correct Word2Vec save path with filename
    w2v_model.save(WORD2VEC_MODEL_PATH)
    print(f"✅ Word2Vec Model saved to: {WORD2VEC_MODEL_PATH}")
    
    X_train, X_test, y_train, y_test = vectorize_text(
        w2v_texts, labels,
        is_TFIDF=False,
        word2vec_model=w2v_model
    )
    
    w2v_log_model = train_model(X_train, y_train, WORD2VEC_MODEL_PATH)
    
    evaluate_model(
        w2v_log_model, X_test, y_test,
        model_name="Word2Vec + Logistic Regression"
    )