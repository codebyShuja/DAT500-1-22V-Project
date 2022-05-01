import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from  sklearn.model_selection import train_test_split

from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import roc_auc_score


dataset = pd.read_csv('pkr.csv', na_filter=False)
df = dataset[:10000]

df = df[df['rating'] != 3]
df['Positively Rated'] = np.where(df['rating'] > 3, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'],df['Positively Rated'], random_state=0)
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))
print('Logistic Regression Using CountVectorizing:> Accuracy Score:: ', roc_auc_score(y_test, predictions))
