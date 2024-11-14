
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D,Activation,Lambda,Embedding, Bidirectional
from keras.layers import LSTM, GRU, SimpleRNN
import tensorflow.keras.layers as L

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Solar Prediction/Book1.csv')
df

df['Daily Power Generation'].value_counts()

data = pd.DataFrame(list(df['Daily Power Generation']), columns=['Daily Power Generation'])
data

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))
data_scaled = scalar.fit_transform(data)

data_scaled.shape

timestep = 4
X= []
Y=[]
for i in range(len(data_scaled)- (timestep)):
    X.append(data_scaled[i:i+timestep])
    Y.append(data_scaled[i+timestep])

X=np.asanyarray(X)
Y=np.asanyarray(Y)

k = 231
Xtrain = X[:k,:,:]
Xtest = X[k:,:,:]
Ytrain = Y[:k]
Ytest= Y[k:]

X.shape

Xtrain.shape

Xtest.shape

from tensorflow.keras.layers import Dense,RepeatVector, LSTM, Dropout
from keras.callbacks import CSVLogger
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

lstm_model = Sequential()
lstm_model.add(L.LSTM(100, input_shape=(4, 1), return_sequences=True))
lstm_model.add(L.LSTM(60, activation='relu', return_sequences=True))
lstm_model.add(L.LSTM(10, activation='relu'))
lstm_model.add(L.RepeatVector(1))
lstm_model.add(L.Dense(100, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(L.Dense(100, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(L.Dense(10))

lstm_model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mse', 'mae'])
csv_logger = CSVLogger('/content/drive/MyDrive/Solar Prediction/Results_AE/encoder.csv',separator=',', append=False)
history = lstm_model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=20, batch_size=16, verbose=1, shuffle =True, callbacks=[csv_logger])

print(lstm_model.summary())

lstm_model.save("./regressor.hdf5")

from tensorflow.keras.utils import plot_model
plot_model(lstm_model, to_file='/content/drive/MyDrive/Solar Prediction/Results_AE/model1.png')

predict1 = lstm_model.predict(Xtest)

predict1

import numpy as np

# Reshape predict1 to 2D
predict1_reshaped = predict1.reshape(predict1.shape[0], -1)

# Apply inverse_transform
predict2 = scalar.inverse_transform(predict1_reshaped)

# predict2 = scalar.inverse_transform(predict1)

Ytesting1 = scalar.inverse_transform(Ytest)

predict2.ndim

plt.figure(figsize=(10,6))
plt.plot(Ytesting1 , 'blue', linewidth=5)
plt.plot(predict2,'r' , linewidth=4)
plt.legend(('Test','Predicted'))
plt.show()

df1 = pd.DataFrame(np.transpose(predict2)).T
df1.to_excel(excel_writer = '/content/drive/MyDrive/Solar Prediction/Results_AE/predicted.xlsx')

df2 = pd.DataFrame(np.transpose(Ytesting1)).T
df2.to_excel(excel_writer = '/content/drive/MyDrive/Solar Prediction/Results_AE/test.xlsx')

k = 0
Xtrain1 = X[:k,:,:]
Xtest1 = X[k:,:,:]
Ytrain1 = Y[:k]
Ytest1= Y[k:]

predict2 = lstm_model.predict(Xtest1)

import numpy as np

# Reshape predict1 to 2D
predict2_reshaped = predict2.reshape(predict2.shape[0], -1)

# Apply inverse_transform
predict3 = scalar.inverse_transform(predict2_reshaped)

# predict3 = scalar.inverse_transform(predict2)

Ytesting1 = scalar.inverse_transform(Ytest)

plt.figure(figsize=(10,6))
plt.plot(Ytesting1 , 'blue', linewidth=5)
plt.plot(predict3,'green' , linewidth=4)
plt.legend(('Test','Predicted'))
plt.show()

df2 = pd.DataFrame(np.transpose(predict3)).T
df2.to_excel(excel_writer = '/content/drive/MyDrive/Solar Prediction/Results_AE/oneyearpred.xlsx')

from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error

# Assuming Ytest is of shape (n_samples,) and yhat_probs is of shape (n_samples, n_features)
yhat_probs = predict1

# Taking the mean of predictions across features (axis=2 if yhat_probs has 2 axes after first mean operation)
yhat_probs_mean = yhat_probs.mean(axis=1)  # This reduces to shape (56, 10)

# Further reduce to a single mean prediction per sample
yhat_probs_final = yhat_probs_mean.mean(axis=1)  # This should reduce it to (56,)

# Ensure Ytest is a 1D array to match yhat_probs_final
Ytest = Ytest.flatten()

# # Print the shapes for debugging
# print(f'Shape of Ytest: {Ytest.shape}')
# print(f'Shape of yhat_probs_final: {yhat_probs_final.shape}')

# Calculate the errors using the final mean predictions
var2 = max_error(Ytest, yhat_probs_final)
print(f'Max Error: {var2}')

mse = mean_squared_error(Ytest, yhat_probs_final)
print(f'Mean Squared Error: {mse}')

mae = mean_absolute_error(Ytest, yhat_probs_final)
print(f'Mean Absolute Error: {mae}')