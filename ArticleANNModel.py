import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import bokeh
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense, GRU
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
from keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping

class annModels(object):


    def __init__(self, neurons,
                 inputTrainData, outputModel,
                 optimizer='adam', loss='mae',
                 modelType='LSTM'):

        self.neurons = neurons
        self.inputTrainData = inputTrainData
        self.outputModel = outputModel
        self.optimizer = optimizer
        self.loss = loss
        self.modelType = modelType
        self.model = None
        self.callbackConfiguration = {}

    def _callbacks(self):
        self.callbackConfiguration['EarlyStopping'] = EarlyStopping(monitor='val_loss',
                                                      patience=5, verbose=1)
        return [self.callbackConfiguration['EarlyStopping']]
    def build(self):
        from keras.models import Sequential
        if self.modelType == 'LSTM':
            self.model = Sequential()
            self.model.add(LSTM(self.neurons,
                           input_shape=(self.inputTrainData.shape[1],
                           self.inputTrainData.shape[2])))
            self.model.add(Dense(self.outputModel))
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer)
            return self.model
        elif self.modelType == 'GRU':
            self.model = Sequential()
            self.model.add(LSTM(self.neurons,
                           input_shape=(self.inputTrainData.shape[1],
                           self.inputTrainData.shape[2])))
            self.model.add(Dense(self.outputModel))
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer)
            return self.model
        elif self.modelType == 'RBF':
            return "Implementar...."

        elif self.modelType == 'MLP':
            return "William vai implementar. hehehe"

    def fit(self):
        return "Vou implementar ainda"

    def plotLossTrainAndValidLoss(self):
        return "Vou implementar ainda esse"

    def metricsToEvaluteModel(self):
        return "Implementar metricas do artigo: Bouktif et al 2019."

if __name__ == '__main__':
    import numpy as np
    print()
    model = annModels(50, np.array([[[10, 20, 30]]]), outputModel=1,
                      optimizer='adam', loss='mae', modelType='RBF')
    print(model.build())
