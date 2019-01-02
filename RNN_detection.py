import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#configuring keras
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, GaussianDropout, LSTM, Bidirectional
import data_processing as dp

# NOTE: comment the next line if running more than once back-to-back
dp.save_data_to_3d('./dataset/EEG_data_178steps_12000samples_5classes.csv', binary=1, augment=0)
eeg_data, labels = dp.load_data()

inputs = Input(shape=(178, 1))
x = GaussianDropout(0.1)(inputs)	#generating noise during training for better generalization
x = LSTM(60)(x)
x = Dropout(0.2)(x)
x = Dense(20)(x)
x = Dropout(0.2)(x)
predictions = Dense(units=2, activation="sigmoid")(x)

model = Model(inputs, predictions)

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(patience=10)
tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(os.path.basename(__file__)[:-3]), histogram_freq=1, write_graph=True, write_images=True)

model.fit(eeg_data, labels, epochs=200, verbose=1, validation_split=0.1, callbacks=[es, tensorboard])
model.save("{}".format(os.path.basename(__file__)[:-3]))
