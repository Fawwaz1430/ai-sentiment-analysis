from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset (2 kategori biar simpel)
categories = ['rec.sport.baseball', 'sci.med']
data = fetch_20newsgroups(categories=categories)

X = data.data
y = data.target

# 2. Ubah teks jadi angka
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# 3. Split data (training & testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# 4. Train model AI
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)

# 6. Tes manual
sample_text = ["This treatment really helped me recover faster"]
sample_vector = vectorizer.transform(sample_text)
prediction = model.predict(sample_vector)

if prediction[0] == 0:
    print("Prediction: Baseball-related (Negative class)")
else:
    print("Prediction: Medical-related (Positive class)")
