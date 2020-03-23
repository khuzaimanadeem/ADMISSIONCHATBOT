import pandas as pd
import nltk
import numpy as np
import re
import heapq

pd.set_option('display.max_colwidth', -1)

file = pd.ExcelFile('Dataset_final.xlsx')

reader1 = pd.read_excel(file, usecols='B')
reader2 = pd.read_excel(file, usecols='A')
Y = reader1
X = reader2

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

print(X_sentence_vectors)


"""
from sklearn.model_selection import train_test_split
lookback = 1
X_train, X_test, y_train, y_test = train_test_split(X_sentence_vectors, Y_sentence_vectors, test_size = 0.2)

from keras import Sequential
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(units=6, input_shape=(1 , lookback) , return_sequences = True))
model.add(LSTM(units=6, return_sequences=True))
model.add(LSTM(units=6, return_sequences=True))
model.add(LSTM(units=1, return_sequences=True, name='output'))
model.compile(loss='cosine_proximity', optimizer='sgd', metrics = ['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test,y_test) , epochs=100, batch_size=1 , verbose=2)
"""
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
sc = MinMaxScaler()
X_train = np.reshape(X_train, (-1, 1))
Y_train = np.reshape(Y_train, (-1, 1))
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train)

X_train = np.reshape(X_train, (384, 1, 1))

regress = Sequential()

regress.add(LSTM((1), batch_input_shape=(None, 1), return_sequences='false'))

regress.compile(optimizer='adam', loss='mean_squared_error')
regress.summary()
#regress.fit(X_train, Y_train, batch_size=10, epochs=50, validation_data=(X_test, Y_test))

#inputs = X_test
#inputs = np.reshape(inputs, (-1, 1))
#inputs = sc.transform(inputs)
#inputs = np.reshape(inputs, (165, 1, 1))
#Y_pred = regress.predict(inputs)
#Y_pred = sc.inverse_transform(Y_pred)

#plt.figure
#plt.plot(Y_test, color='red', label='Real Web View')
#plt.plot(Y_pred, color='blue', label='Predicted Web View')
#plt.title('Web View Forecasting')
#plt.xlabel('Number of Days from Start')
#plt.ylabel('Web View')
#plt.legend()
#plt.show()

"""
