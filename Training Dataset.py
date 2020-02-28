import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
file = pd.ExcelFile('Dataset_final.xlsx')
reader1 = pd.read_excel(file, usecols='B')
reader2 = pd.read_excel(file, usecols='A')

Y = reader1
X = reader2
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

