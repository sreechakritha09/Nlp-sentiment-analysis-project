# Nlp-sentiment-analysis-project
#this project is about sentiment anakysis on product reviews on flipkart
#the dataset contains, reviews and corresponding ratings , tis was taken from kaggle
#the code essentially otputs a confsion matrix, predicted values as well as accuracy, precision, recall, f1 score paramaters
# we shall use this info to judge how well the model works in predicting correctly
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel file
data = pd.read_csv('/content/data.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['processed_review'] = data['review'].apply(preprocess_text)

# Convert ratings to sentiment labels
data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x > 4 else 'neutral' if 3 <= x <= 4 else 'negative')

# Remove neutral reviews if needed
# data = data[data['sentiment'] != 'neutral']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_review'], data['sentiment'], test_size=0.2, random_state=42
)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualizing the confusion matrix
labels = ['negative', 'neutral', 'positive']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Count the number of positive, neutral, and negative reviews predicted by the model
positive_reviews = sum(y_pred == 'positive')
neutral_reviews = sum(y_pred == 'neutral')
negative_reviews = sum(y_pred == 'negative')

print(f"Predicted Positive Reviews: {positive_reviews}")
print(f"Predicted Neutral Reviews: {neutral_reviews}")
print(f"Predicted Negative Reviews: {negative_reviews}")
