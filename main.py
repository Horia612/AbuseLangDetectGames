import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import re
import nltk

# Load datasets
train_data = pd.read_csv('CONDA_train.csv')
valid_data = pd.read_csv('CONDA_valid.csv')
test_data = pd.read_csv('CONDA_test.csv')

# Download NLTK stopwords data
nltk.download('stopwords')

# Preprocess text: remove special characters, lowercase, and tokenize
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    else:
        # Handle non-string input (e.g., float, int, etc.)
        return []

# Remove stopwords
stop_words = set(stopwords.words('english'))
train_data['processed_utterance'] = train_data['utterance'].apply(lambda x: ' '.join([word for word in preprocess_text(x) if word not in stop_words]))
valid_data['processed_utterance'] = valid_data['utterance'].apply(lambda x: ' '.join([word for word in preprocess_text(x) if word not in stop_words]))
test_data['processed_utterance'] = test_data['utterance'].apply(lambda x: ' '.join([word for word in preprocess_text(x) if word not in stop_words]))

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train = tfidf_vectorizer.fit_transform(train_data['processed_utterance'])
X_valid = tfidf_vectorizer.transform(valid_data['processed_utterance'])
X_test = tfidf_vectorizer.transform(test_data['processed_utterance'])

y_train = train_data['intentClass']
y_valid = valid_data['intentClass']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions on validation set
y_valid_pred = model.predict(X_valid)

# Evaluate the model on validation set
print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
print(classification_report(y_valid, y_valid_pred))

# Predictions on test set
y_test_pred = model.predict(X_test)

# Save predictions to a CSV file (you can modify this part)
test_data['predicted_intentClass'] = y_test_pred
test_data.to_csv('CONDA_test_predictions.csv', index=False)
