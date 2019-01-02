import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#configuring keras
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, GaussianDropout, BatchNormalization
import data_processing as dp
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import matplotlib.pyplot as plt

# NOTE: comment out next line if running more than once back-to-back
dp.save_data('./dataset/EEG_data_178steps_12000samples_5classes.csv', binary=1, augment=0)
eeg_data, labels = dp.load_data()

X_train = list([])
X_anomaly = list([])

for i in range(eeg_data.shape[0]): 
	if labels[i] != 0: 
		X_train.append(eeg_data[i])
	else: 
		X_anomaly.append(eeg_data[i])

X_train = np.asarray(X_train)
X_anomaly = np.asarray(X_anomaly)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_anomaly = X_anomaly.reshape(X_anomaly.shape[0], X_anomaly.shape[1])

#Normalizing the input
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_anomaly = scaler.fit_transform(X_anomaly)
X_train, X_test = X_train[0:8500], X_train[8500:]

inputs = Input(shape=(178,))
x = GaussianDropout(0.05)(inputs)
x = Dense(80, activation='tanh')(x)
x = BatchNormalization()(x)
x = Dense(40, activation='tanh')(x)
x = BatchNormalization()(x)
x = Dense(20, activation='tanh')(x)
x = BatchNormalization()(x)
x = Dense(40, activation='tanh')(x)
x = BatchNormalization()(x)
x = Dense(80, activation='tanh')(x)
x = BatchNormalization()(x)
predictions = Dense(178, activation='tanh')(x)

autoEncoder = Model(inputs, predictions)
autoEncoder.summary()

adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoEncoder.compile(adam, loss='mse')	

tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(os.path.basename(__file__)[:-3]), histogram_freq=1, write_graph=True, write_images=True)

autoEncoder.fit(X_train, X_train, epochs=50, verbose=1, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[tensorboard])
autoEncoder.save("{}".format(os.path.basename(__file__)[:-3]))

#autoEncoder = load_model("{}".format(os.path.basename(__file__)[:-3]))
#autoEncoder.summary()

THRESHOLD = 0.03

predictions = autoEncoder.predict(X_anomaly)
mseSeizure = np.mean(np.power(X_anomaly - predictions, 2), axis=1)
print ("false negatives: " + str(np.sum(mseSeizure < THRESHOLD)) + "out of " + str(mseSeizure.shape[0]))

predictions = autoEncoder.predict(X_test)
mseNormal = np.mean(np.power(X_test - predictions, 2), axis=1)
print ("false positives: " + str(np.sum(mseNormal > THRESHOLD)) + "out of " + str(mseNormal.shape[0]))

plt.hist([mseSeizure, mseNormal], color=["red", "blue"])
plt.xlabel("MSE")
plt.ylabel("# of samples")
plt.legend()
plt.show()
