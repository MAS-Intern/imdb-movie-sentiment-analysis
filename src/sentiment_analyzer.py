"""
IMDB Movie Sentiment Analysis using NLP and Machine Learning
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class SentimentAnalyzer:
    """IMDB Movie Review Sentiment Analyzer"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stop words
        self.stop_words = self.stop_words - {'not', 'no', 'nor', 'neither'}
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        self.best_model = None
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, text_column='review'):
        """Preprocess entire dataset"""
        df['cleaned_review'] = df[text_column].apply(self.preprocess_text)
        return df
    
    def prepare_features(self, reviews, labels=None):
        """Convert text to features using TF-IDF"""
        X = self.vectorizer.fit_transform(reviews) if labels is not None else self.vectorizer.transform(reviews)
        return X
    
    def train_model(self, data_path, text_column='review', label_column='sentiment', 
                    test_size=0.2, model_name='Logistic Regression'):
        """
        Train sentiment classification model
        
        Parameters:
        data_path: Path to CSV file with reviews
        text_column: Name of column containing review text
        label_column: Name of column containing sentiment labels
        test_size: Proportion of data for testing
        model_name: Name of model to train
        """
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df = self.preprocess_dataset(df, text_column)
        
        # Encode labels (assuming 'positive' = 1, 'negative' = 0)
        y = (df[label_column] == 'positive').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'], y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create feature vectors
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train selected model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.models.keys())}")
        
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Store best model
        self.best_model = {
            'name': model_name,
            'model': model,
            'accuracy': accuracy,
            'vectorizer': self.vectorizer
        }
        self.is_trained = True
        
        return accuracy
    
    def predict(self, review_text):
        """
        Predict sentiment for a single review
        
        Parameters:
        review_text: str - Movie review text
        
        Returns:
        dict - Prediction with label and confidence
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Preprocess
        cleaned = self.preprocess_text(review_text)
        X = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.best_model['model'].predict(X)[0]
        probabilities = self.best_model['model'].predict_proba(X)[0]
        
        label = 'Positive' if prediction == 1 else 'Negative'
        confidence = probabilities[prediction]
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'Positive': probabilities[1],
                'Negative': probabilities[0]
            }
        }
    
    def predict_batch(self, reviews):
        """Predict sentiment for multiple reviews"""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        cleaned = [self.preprocess_text(r) for r in reviews]
        X = self.vectorizer.transform(cleaned)
        
        predictions = self.best_model['model'].predict(X)
        probabilities = self.best_model['model'].predict_proba(X)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'label': 'Positive' if pred == 1 else 'Negative',
                'confidence': probabilities[i][pred]
            })
        
        return results
    
    def evaluate(self):
        """Evaluate the best trained model"""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        return {
            'model_name': self.best_model['name'],
            'accuracy': self.best_model['accuracy']
        }
    
    def get_feature_importance(self, top_n=20):
        """Get most important features (words) for sentiment classification"""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if self.best_model['name'] == 'Logistic Regression':
            coefficients = self.best_model['model'].coef_[0]
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Top positive words
            top_positive_idx = coefficients.argsort()[-top_n:]
            top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx[::-1]]
            
            # Top negative words
            top_negative_idx = coefficients.argsort()[:top_n]
            top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]
            
            return {
                'top_positive_words': top_positive,
                'top_negative_words': top_negative
            }
        
        return {}
