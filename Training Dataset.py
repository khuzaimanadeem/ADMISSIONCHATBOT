import pandas as pd
import nltk
import numpy as np
import re
import heapq

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
one = OneHotEncoder()
pd.set_option('display.max_colwidth', -1)
data = pd.read_csv('Dataset_final2.csv')

X = data['QUESTIONS']

Y = le.fit_transform(data['INTENTS'])
Y = one.fit_transform(Y.reshape(-1,1)).toarray()



X_corpus = nltk.sent_tokenize(X.to_string())

for i in range(len(X_corpus)):
    X_corpus[i] = X_corpus[i].lower()
    X_corpus[i] = re.sub(r'\W', ' ', X_corpus[i])
    X_corpus[i] = re.sub(r'\s+', ' ', X_corpus[i])

X_word_freq = {}
for sentence in X_corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in X_word_freq.keys():
            X_word_freq[token] = 1
        else:
            X_word_freq[token] += 1


X_most_freq = heapq.nlargest(200, X_word_freq, key=X_word_freq.get)

X_sentence_vectors = []
for sentence in X_corpus:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in X_most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    X_sentence_vectors.append(sent_vec)

X_sentence_vectors = np.asarray(X_sentence_vectors)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_sentence_vectors, Y, test_size = 0.2)

from keras import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=6, input_shape=(485,200), return_sequences = True))
model.add(LSTM(units=6,return_sequences=True))
model.add(Dense(64, activation='relu'))

model.add(LSTM(units=1, return_sequences=True, name='output'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=100, verbose=2 , validation_data=(X_test,y_test))
