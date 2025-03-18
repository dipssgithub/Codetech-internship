#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset (Replace 'customer_reviews.csv' with your actual dataset file)
df = pd.read_csv(r"C:\Users\Administrator\Downloads\customer_reviews.csv")  # Ensure dataset has 'review' & 'sentiment' columns

# Display first few rows
display(df.head())

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid overfitting
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Test with New Reviews (Optional)
def predict_sentiment(new_reviews):
    processed_reviews = [preprocess_text(review) for review in new_reviews]
    vectorized_reviews = tfidf_vectorizer.transform(processed_reviews)
    predictions = model.predict(vectorized_reviews)
    return predictions

# Example Usage
new_reviews = ["I love this product!", "Worst experience ever."]
predictions = predict_sentiment(new_reviews)
print(list(zip(new_reviews, predictions)))


# In[ ]:




