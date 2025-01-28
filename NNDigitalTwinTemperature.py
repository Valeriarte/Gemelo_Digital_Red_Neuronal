#!/usr/bin/env python
# coding: utf-8

# # Class to implement neural network digital twin model for 2D sensors
# 
# This class is to be used to create digital twin models for 2D ($y=f(x)$) type sensors. The model inputs are the $x$ and $y$ datasets obtained from the datasheet or a measured data from the sensor, and it returns the trained neural networks for the functions $y=f(x)$ and $x=f⁻1(y)$. If it's wanted, you can save the trained model to be used when its needed. The UML class implementation is the following:

# In[ ]:


# libraries for the files in google drive
from pydrive.auth import GoogleAuth
from google.colab import drive
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import matplotlib.pyplot as plt
import os
from pylab import rcParams
import statsmodels.api as sm
import matplotlib.dates as mdates
import seaborn as sns


# In[ ]:


import numpy as np
import keras as kr
import plotly.graph_objects as go
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[ ]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1QJjkWpxAIsobw26_BlTYTbsBdHCuSSfD' #<-- You add in here the id from you google drive file, you can find it


# In[ ]:


# Crear objeto descargable
download = drive.CreateFile({'id': file_id})

# Descargar archivo en Colab
download.GetContentFile('DatosFinalFinal.xlsx')


# In[ ]:


data = pd.read_excel('DatosFinalFinal.xlsx')
# Mostrar los primeros registros
print(data.head())


# In[ ]:


tiempo = data['Tiempo'].values.tolist()
temperatura = data['Temperatura'].values.tolist()
xData = np.array(tiempo)
yData = np.array(temperatura)


# In[ ]:


class XNNSensor:

    def __init__(self, x, y):
        self.funXYModel = None
        self.invfunYXModel = None
        self.xData = x
        self.yData = y

    def funXY(self, x):
        return self.funXYModel.predict(x)

    def getfunXYModel(self,fActivation):
        try:
            # obtain f in y = f(x)
            self.funXYModel = Model(self.xData, self.yData)
            self.funXYModel.train(fActivation)

        except Exception as e:
            print("[X] Training error:",e)


    def invfunYX(self, y):
        return self.invfunYXModel.predict(y)

    def getinvfunYXModel(self,fActivation):
        try:
            #Training dataset
            xInput = np.linspace(self.xData[0], self.xData[-1], 1000)
            yOutput = np.array([self.funXY(i) for i in xInput ])

            # obtain f⁻1 in x=f⁻1(y)
            self.invfunYXModel = Model(yOutput,xInput)
            self.invfunYXModel.train(fActivation)

        except Exception as e:
            print("[X] Training error: ",e)

    def saveModels(self, nameModel):
        try:
            self.funXYModel.model.save(nameModel + '_funXY.h5')
            self.invfunYXModel.model.save(nameModel + '_invfunYX.h5')
        except Exception as e:
            print('Error while saving y=f(x) and x=f⁻1(y) models: ', e)


# In[ ]:


class Model:

    def __init__(self,xValues,yValues):
        self.xValues = xValues
        self.yValues = yValues
        self.model = None

    def train(self,activation='relu'):
        try:
            self.model = kr.Sequential()
            self.model.add(kr.layers.Dense(32, activation = activation))
            self.model.add(kr.layers.Dense(32, activation = activation))
            self.model.add(kr.layers.Dense(1))

            epochs = 1000
            loss = "mse"
            self.model.compile(optimizer='adam',
                          loss=loss,
                          metrics=['mae'], #Mean Absolute Error
                        )
            self.xValues = self.xValues.reshape(-1, 1)
            self.yValues = self.yValues.reshape(-1, 1)

            history = self.model.fit(self.xValues, self.yValues,
                                shuffle=True,
                                epochs=epochs,
                                batch_size=20,
                                verbose=0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(0, epochs, 1),
                                     y=history.history['mae'],
                                     mode='lines',
                                     name=f'Training MAE',
                                     marker_size=3,
                                     line_color='orange'))
            fig.add_trace(go.Scatter(x=np.arange(0, epochs, 1),
                                     y=history.history['val_mae'],
                                     mode='lines',
                                     name=f'Validation MAE',
                                     line_color='grey'))
            fig.update_layout(
                              title="Network training",
                              xaxis_title="Epochs",
                              yaxis_title=f"Mean Absolute Error")
            fig.update_layout({'plot_bgcolor': '#f2f8fd' ,
                               'paper_bgcolor': 'white',},
                               template='plotly_white')

        except Exception as e:
            print("[X]Error when computing model: ",e)

    def predict(self, inputVal):
        try:

            outputVal = self.model.predict([inputVal], verbose=False)
            return outputVal[0][0]

        except Exception as e:
            print("[X] Predict error: ", e)


# ## Use case example
# 
# ### Digital twin for Temperature control
# 

# In[ ]:


#Se crean y entrenan los modelos sensor de masa
masSensor = XNNSensor(xData, yData)
masSensor.getfunXYModel('relu')
masSensor.getinvfunYXModel('relu')

#Samples from datasheet
massTrain10, voltTrain10 = masSensor.xData, masSensor.yData

#Interpolating samples
massTest = np.linspace(0, 1000, 20) #mass between 0 to 1000 grams, 20 samples

#Model prediction
voltPredict10 = [masSensor.funXY(i) for i in massTest]

plt.plot(massTrain10, voltTrain10,'o')
plt.plot(massTest, voltPredict10)
plt.xlabel('Tiempo')
plt.ylabel('Temperatura')
plt.title('Modelo de Temperature control')
plt.legend(['Datos', 'Modelo'])
plt.savefig("Temperature_control")


# In[ ]:


#Repeating the previous steps to replicate the inverse function

#New current samples, obtained from the previous model
voltTrain10_, massTrain10_ = np.array(voltPredict10, dtype=float), massTest

#Model prediction
massPredict10_ = [masSensor.invfunYX(i) for i in voltTrain10_]

plt.plot(voltTrain10_, massTrain10_, 'o')
plt.plot(voltTrain10_, massPredict10_, 'r')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura')
plt.title(r"inverse digital twin, Temperature_control")
plt.legend(["Sample","Model"])


# In[ ]:


# saving y=f(x) and x=f⁻1(y) functions in h5 files
masSensor.saveModels('Temperature_control')


# In[ ]:


# Example of digital twin simulation use, simulating the mass read by the sensor in a simulation scenario
# grados para el sensor
x = 35
# read obtained by the sensor
masSensor.invfunYX(float(masSensor.funXY(x)))


# In[ ]:


import os
print(os.getcwd())


# In[ ]:


get_ipython().system('ls /content')

