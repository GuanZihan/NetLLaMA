import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
# from collections import Counter
# from textblob import TextBlob

# load the dataset
df = pd.read_csv('/Users/yanxi/Documents/GitHub/NetLLaMA/datasets/TeleQnA.csv') 

# (1) Vectorize the questions using TF-IDF and reduce dimensionality with PCA
vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = vectorizer.fit_transform(df['question']).toarray()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA of Question Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# (2) Analyze the distribution of question categories

# Get the count for each category
category_counts = df['category'].value_counts()

# Convert to percentages
category_percentages = category_counts / category_counts.sum() * 100

# Draw the pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_percentages, labels=category_percentages.index, autopct='%1.1f%%', startangle=140,
        pctdistance=0.85)
plt.title('Distribution of Question Categories in Dataset')
plt.tight_layout()
plt.show()
# (3) Analyze the distribution of question lengths
# question_lengths = df['question'].apply(len)

# plt.figure(figsize=(10, 6))
# # plt.hist(question_lengths, bins=30, color='skyblue')
# plt.hist(question_lengths, bins=30, color='skyblue', edgecolor='black', linewidth=1.5)
# plt.title('Distribution of Question Lengths')
# plt.xlabel('Length of Question (characters)')
# plt.ylabel('Frequency')
# plt.show()
# Define a function to extract the length of the question text
def get_question_length(text):
    # The question is up to the third comma
    question_text = text.split(',', 3)[2] if len(text.split(',', 3)) > 2 else text
    return len(question_text)

# Apply the function to the 'question' column to get the length of each question
df['question_length'] = df['question'].apply(get_question_length)

# Plot the distribution of question lengths
plt.figure(figsize=(10, 6))
plt.hist(df['question_length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Question Lengths')
plt.xlabel('Length of Question (characters)')
plt.ylabel('Frequency')
plt.show()
