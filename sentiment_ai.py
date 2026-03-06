# AI Sentiment Analysis with Accuracy
# Author: Fawwaz Khatami

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Training data
train_texts = [
    "I love this product",
    "This is amazing",
    "I am very happy",
    "I hate this",
    "This is terrible",
    "Very bad experience"
]

train_labels = [
    "positive", "positive", "positive",
    "negative", "negative", "negative"
]

# Testing data
test_texts = [
    "I am happy",
    "This product is bad",
    "Amazing experience",
    "I hate it"
]

test_labels = [
    "positive", "negative", "positive", "negative"
]

# Vectorization
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Model training
model = MultinomialNB()
model.fit(X_train, train_labels)

# Prediction & accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)

print("AI Sentiment Model Ready")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Interactive input
user_text = input("Enter a sentence: ")
user_vector = vectorizer.transform([user_text])
result = model.predict(user_vector)

print("Sentiment prediction:", result[0])
