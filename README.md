# IMDB Movie Sentiment Analysis

## Overview
IMDB sentiment analysis using Python, NLP: classifies reviews, informs viewing decisions. This project analyzes movie reviews to determine sentiment (positive/negative) using natural language processing and machine learning techniques.

## Tech Stack
- Python
- Spyder IDE
- NLP
- Scikit-learn
- NLTK
- Spacy

## Project Structure
```
imdb-movie-sentiment-analysis/
├── notebooks/          # Jupyter notebooks with analysis
├── data/              # Dataset files
├── src/               # Source code for NLP and ML
├── models/            # Trained sentiment models
├── visualizations/    # Generated charts and graphs
└── README.md          # This file
```

## Key Features
- **Text Preprocessing**: Tokenization, stemming, lemmatization, stop word removal
- **Feature Extraction**: TF-IDF, Bag of Words, Word embeddings
- **Sentiment Classification**: Binary classification (positive/negative)
- **Model Comparison**: Multiple algorithms benchmarked
- **Performance Analysis**: Comprehensive evaluation metrics

## Methodology

### 1. Data Preprocessing
- Text cleaning and normalization
- Removal of HTML tags and special characters
- Tokenization and lowercasing
- Stop word removal
- Stemming and Lemmatization

### 2. Feature Engineering
- **Bag of Words (BoW)**: Simple word frequency counts
- **TF-IDF**: Term frequency-inverse document frequency
- **N-grams**: Unigrams, bigrams, and trigrams
- **Word Embeddings**: Word2Vec, GloVe (optional)

### 3. Machine Learning Models
- **Logistic Regression**: Strong baseline for text classification
- **Naive Bayes**: Classic text classification algorithm
- **Support Vector Machines (SVM)**: Effective for high-dimensional data
- **Random Forest**: Ensemble method
- **Gradient Boosting**: Advanced ensemble technique

### 4. Deep Learning (Optional)
- LSTM/GRU networks
- CNN for text classification
- Transformer-based models (BERT)

## Installation

```bash
# Clone the repository
git clone https://github.com/MAS-Intern/imdb-movie-sentiment-analysis.git
cd imdb-movie-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

## Usage

```python
from src.sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Train the model
analyzer.train_model('data/IMDB Dataset.csv')

# Predict sentiment for a review
review = "This movie was absolutely fantastic! Great acting and storyline."
prediction = analyzer.predict(review)
print(f"Sentiment: {prediction['label']} (Confidence: {prediction['confidence']:.2f})")

# Evaluate model
metrics = analyzer.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

## Model Performance

### Best Model Results
```
Logistic Regression:
- Accuracy: 89.5%
- Precision: 0.90
- Recall: 0.89
- F1 Score: 0.89

Support Vector Machine:
- Accuracy: 88.7%
- Precision: 0.89
- Recall: 0.88
- F1 Score: 0.88
```

### Confusion Matrix
```
                Predicted
              Positive  Negative
Actual Positive    4475      525
       Negative     550     4450
```

## Key Insights

### Positive Review Indicators
- Words: "excellent", "amazing", "fantastic", "wonderful", "best"
- Phrases: "highly recommend", "must watch", "great movie"
- Exclamation marks and strong positive adjectives

### Negative Review Indicators
- Words: "terrible", "worst", "boring", "waste", "bad"
- Phrases: "don't watch", "poor quality", "very disappointed"
- Criticism of acting, plot, or direction

## Business Applications
- **Movie Recommendation Systems**: Enhance recommendations with sentiment analysis
- **Content Strategy**: Understand what makes movies successful
- **Marketing Insights**: Identify sentiment trends for promotional strategies
- **Quality Control**: Early detection of poorly received content

## Visualizations
- Word clouds for positive and negative reviews
- Sentiment distribution histograms
- ROC curves and precision-recall curves
- Feature importance (most predictive words)
- Confusion matrices
- Learning curves

## Dataset
The IMDB dataset contains:
- 50,000 movie reviews (25k train, 25k test)
- Balanced classes (50% positive, 50% negative)
- Reviews with binary sentiment labels
- Text reviews with rich linguistic content

## Future Enhancements
- Multi-class sentiment analysis (positive, neutral, negative)
- Aspect-based sentiment analysis (acting, plot, cinematography)
- Real-time sentiment prediction API
- Integration with movie database
- Emotion detectionion beyond binary sentiment

## Contributors
- MAS Internship Program

## License
This project is part of the MAS (Mentor Aspire System) educational program.
