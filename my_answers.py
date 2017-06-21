import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # Data length and window size
    P = len(series)
    T = window_size

    # Input and output pairs
    X = [series[i:i + T] for i in range(0, P - T)]
    y = [series[i] for i in range(T, P)]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))  # First layer
    model.add(Dense(1))  # Second layer

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


### TODO: list all unique characters in the text and remove any non-english ones
import re
def clean_text(text):
    # find all unique characters in the text
    chars = list(set(text))

    # remove as many non-english characters and character sequences as you can 
    rx = re.compile('[^a-zA-Z0-9!?,;\'\"\(\)]')
    text = rx.sub(' ', text)

    # shorten any extra dead space created above
    return text.replace('  ', ' ')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # Data length and window size
    P = len(text)
    T = window_size
    S = step_size

    # Input and output pairs
    inputs = [text[i:i + T] for i in range(0, P - T, S)]
    outputs = [text[i] for i in range(T, P, S)]

    return inputs, outputs
