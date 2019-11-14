import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras
import string

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    for n in range(series.shape[0] - window_size): # Number of lists is P - T
        X.append([series[i] for i in range(n, n+window_size)])

    y = [series[n + window_size] for n in range(series.shape[0] - window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape=(window_size, 1)))
    model.add(Dense(units=1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # Get the unique characters in ascii
    ascii_all = string.ascii_letters

    # The unique characters in the text
    unique_chars = ''.join(set(text))

    # If the text is a list convert it to a string first
    if type(text) is list:
        text = ''.join(text)

    # Iterate through all the unique characters
    for c in unique_chars:
        if c not in ascii_all and c not in punctuation:
            text = text.replace(c, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[n:n + window_size] for n in range(0, len(text) - window_size, step_size)]
    outputs = [text[n + window_size] for n in range(0, len(text) - window_size, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    model.add(Dense(units=num_chars))
    model.add(Activation('softmax'))
    return model
