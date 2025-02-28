# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:25:23 2025

@author: randy
"""
#%% Codigo original
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#LSTM SIRVE PARA CONJUNTOS CON TIEMPOS CLAROS, LO QUE PASA EN EL PASADO INFLUYE EN EL FUTURO

# 1 GENERAR DATOS DE UNA ONDA SINUSOIDAL
np.random.seed(42)
time = np.arange (0,100,0.1) #genero 1000 puntos
sin_wave = np.sin(time) + np.random.normal(scale=0.1, size=len(time)) #con esto añadimos ruido

df= pd.DataFrame({"time_idx": np.arange(len(time)), "value": sin_wave})

# 2 CONVERTIR DATOS A FORMATO DE SERIES TEMPORALES
def create_sequences(data, seq_length):
    X, y = [], [] #aqui vamos a guardar las secuencias de entrada en X y los valores a predecir en y
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length]) #estos seran los valores de entrada
        y.append(data[i+seq_length]) #estos son los valores de y, los valores a predecir
    return np.array(X), np.array(y)

seq_length= 35 #esta es mi ventana de tiempo
X, y = create_sequences(df["value"].values, seq_length)

#ahora esto debemos convertir a tensores sde pytorh
X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) #AÑADIMOS UNA DIMENSION A LOS DATOS PARA QUE TENGAN FORMA DE TENSOR
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# 3 DEFINIMOS EL MODELO LSTM
class LSTMModel(nn.Module): #nn.module es la clase base para todo modelo de pytorch
    def __init__(self, input_size =1, hidden_size=90, num_layers=2): #cuando ejecutamos lstm se ejecuta esto como incializador de parametros
        super (LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #esta es la capa LSTM donde definimos sus caracteristicas
        self.fc = nn.Linear(hidden_size, 1) #despues de que hayan salido nuestros datos las queremos convertir en un solo valor (prediccion) tomamos un tensor y lo reduce a tamaño 1
        
    def forward(self, x): #con esta funcion defino como quiero que mis datos fluyan a traves del modelo
        lstm_out, _ = self.lstm(x) #de aqui obtenemos una matriz tridimensional con todas las salidas de LSTM
        return self.fc(lstm_out[:,-1,:]) #aqui pedimos que nos devuelva la salida del ultimo espacio de tiempo

model = LSTMModel() #aqui nos deja un batch size,1

# 4 ENTRENAMIENTO DEL MODELO
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=40
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()#limpiamos en cada iteracion donde el modelo ve los datos sus gradientes para que no haya acumulacion
    output = model(X_train) #metemos los datos de entrenamiento al modelo y sacamos el output
    loss = criterion(output, y_train) #calculo la perdida entre los datos entrenados por el modelo y los comparos con los datos reales
    loss.backward()#aqui calculamos el gradiente con respecto a la perdida y con retropropagacion se ajustan los pesos
    optimizer.step() #aqui actualizamos los pesos del modelo despues de haber recalculado los gradientes
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item()}")
    
# 5 HACEMOS LAS PREDICCIONES
model.eval()
with torch.no_grad(): #desactivo el calculo de gradientes para ahorra memoria
    y_pred = model(X_train).numpy()
    

# 6️⃣ Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(df["time_idx"].values[seq_length:], df["value"].values[seq_length:], label="Real")
plt.plot(df["time_idx"].values[seq_length:], y_pred, label="Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Onda Sinusoidal con LSTM")
plt.show()

#%% VAMOS A APLICAR LA REGLA DEL 80/20
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1️⃣ Generar datos de una onda sinusoidal
np.random.seed(42)
time = np.arange(0, 100, 0.1)  # Genero 1000 puntos
sin_wave = np.sin(time) + np.random.normal(scale=0.1, size=len(time))  # Añadimos ruido

df = pd.DataFrame({"time_idx": np.arange(len(time)), "value": sin_wave})

# 2️⃣ Convertir datos a formato de series temporales
def create_sequences(data, seq_length):
    X, y = [], []  # Aquí guardamos las secuencias de entrada en X y los valores a predecir en y
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Valores de entrada
        y.append(data[i + seq_length])  # Valores a predecir
    return np.array(X), np.array(y)

seq_length = 30  # Ventana temporal
X, y = create_sequences(df["value"].values, seq_length)

# Dividir datos en entrenamiento y validación
train_size = int(len(X) * 0.8)  # 80% entrenamiento, 20% validación
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 3️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 4️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item()}, Pérdida de validación: {val_loss.item()}")

# 5️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    val_pred = model(X_val).numpy()

# 6️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
# Entrenamiento
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], df["value"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
# Validación
plt.plot(df["time_idx"].values[train_size+seq_length:], df["value"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df["time_idx"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Onda Sinusoidal con LSTM")
plt.show()

# 7️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()

#%% #EJEMPLO CON DATOS SINTETICOS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1️⃣ Cargar datos sintéticos
file_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\celia_lozano\TimeSeries\data\Datos_Sint_ticos_de_Onda_Sinusoidal.csv"
df = pd.read_csv(file_path)

# 2️⃣ Convertir datos a formato de series temporales
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 15  # Ventana temporal
X, y = create_sequences(df["value"].values, seq_length)

# Dividir datos en entrenamiento y validación
train_size = int(len(X) * 0.8)  # 80% entrenamiento, 20% validación
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 3️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=90, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 4️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item()}, Pérdida de validación: {val_loss.item()}")

# 5️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    val_pred = model(X_val).numpy()

# 6️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
# Entrenamiento
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], df["value"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
# Validación
plt.plot(df["time_idx"].values[train_size+seq_length:], df["value"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df["time_idx"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Onda Sinusoidal con LSTM")
plt.show()

# 7️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()



#%% MODELO DE ENTRENAMIENTO 1, GUARDADO EN CARPETA DE ENTRENAMIENTO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Ruta donde guardaremos el modelo
model_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\Kevin\series_temp_entrenar_modelo\lstm_model.pth"

# 1️⃣ Generar datos de una onda sinusoidal
np.random.seed(42)
time = np.arange(0, 100, 0.1)  # Genero 1000 puntos
sin_wave = np.sin(time) + np.random.normal(scale=0.1, size=len(time))  # Añadimos ruido

df = pd.DataFrame({"time_idx": np.arange(len(time)), "value": sin_wave})

# 2️⃣ Convertir datos a formato de series temporales
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(df["value"].values, seq_length)

# Dividir datos en entrenamiento y validación
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 3️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 4️⃣ Cargar modelo si existe
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Modelo cargado correctamente desde el disco.")
else:
    print("No se encontró un modelo previo. Se entrenará desde cero.")

# 5️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item()}, Pérdida de validación: {val_loss.item()}")

# 6️⃣ Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# 7️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    val_pred = model(X_val).numpy()

# 8️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], df["value"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
plt.plot(df["time_idx"].values[train_size+seq_length:], df["value"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df["time_idx"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Onda Sinusoidal con LSTM")
plt.show()

# 9️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()



#%% MODELO DE ENTRENAMIENTO 2, GUARDADO EN CARPETA DE ENTRENAMIENTO 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Ruta donde guardaremos el modelo
model_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\Kevin\series_temp_entrenar_modelo\lstm_model.pth"

# 1️⃣ Generar datos de una onda cosenoidal con patrones de temporada
np.random.seed(42)
time = np.arange(0, 700, 0.1)  # Onda más larga (4000 puntos)

# 🔄 Onda con estacionalidad: combinación de varias ondas cosenoidales
seasonal_wave = np.cos(time / 5) + 0.5 * np.cos(time / 20) + 0.2 * np.cos(time / 50)
seasonal_wave += 0.002 * time  # Tendencia suave
seasonal_wave += np.random.normal(scale=0.1, size=len(time))  # Ruido normal

# ⚡ Introducir un shock exógeno (evento único de gran impacto)
shock_index = np.random.randint(len(time) // 2, len(time))  # Evento en la segunda mitad
shock_magnitude = np.random.uniform(-11, 11)  # Puede ser positivo o negativo
seasonal_wave[shock_index:] += shock_magnitude  # Aplica el cambio a partir del evento

# 📊 Crear DataFrame final
df = pd.DataFrame({"time_idx": np.arange(len(time)), "value": seasonal_wave})



# 2️⃣ Convertir datos a formato de series temporales
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(df["value"].values, seq_length)

# Dividir datos en entrenamiento y validación
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 3️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 4️⃣ Cargar modelo si existe
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Modelo cargado correctamente desde el disco.")
else:
    print("No se encontró un modelo previo. Se entrenará desde cero.")

# 5️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item()}, Pérdida de validación: {val_loss.item()}")

# 6️⃣ Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# 7️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    val_pred = model(X_val).numpy()

# 8️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], df["value"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
plt.plot(df["time_idx"].values[train_size+seq_length:], df["value"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df["time_idx"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Onda Sinusoidal con LSTM")
plt.show()

# 9️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()



#%% MODELO DE ENTRENAMIENTO 3: FINANCE, GUARDADO EN CARPETA DE ENTRENAMIENTO


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 📍 Ruta donde guardaremos el modelo
model_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\Kevin\series_temp_entrenar_modelo\lstm_model.pth"

# 📍 Ruta del archivo CSV (Asegúrate de cambiar esto por la ruta real)
file_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\celia_lozano\TimeSeries\data\Bitcoin_Historical_Data.csv"

# 1️⃣ Cargar el CSV con formato correcto
df = pd.read_csv(file_path)

# 2️⃣ Convertir la columna 'Date' a formato datetime
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# 3️⃣ Ordenar los datos por fecha (por si vienen en orden inverso)
df = df.sort_values(by="Date").reset_index(drop=True)

# 4️⃣ Seleccionar solo la columna 'Price' (equivalente a 'Close')
df = df[["Date", "Price"]].rename(columns={"Price": "Close"})
# Convertir 'Close' a float (limpiando posibles errores de formato)
df["Close"] = df["Close"].astype(str).str.replace(",", "").astype(float)


# 🔥 NO normalizamos los datos, trabajamos con valores originales 🔥
from sklearn.preprocessing import MinMaxScaler

# Normalizar los precios entre -1 y 1
scaler = MinMaxScaler(feature_range=(-1, 1))
df["Close"] = scaler.fit_transform(df[["Close"]])

# 5️⃣ Convertir en secuencias para la LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Ventana temporal de 30 días
X, y = create_sequences(df["Close"].values, seq_length)

# 6️⃣ Dividir en entrenamiento (80%) y validación (20%)
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 7️⃣ Convertir a tensores para PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (batch, seq_length, input_size)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 8️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 🔟 Cargar modelo si existe
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("✅ Modelo cargado correctamente desde el disco.")
else:
    print("⚠️ No se encontró un modelo previo. Se entrenará desde cero.")

# 1️⃣1️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item():.6f}, Pérdida de validación: {val_loss.item():.6f}")

# 1️⃣2️⃣ Guardar modelo entrenado
#torch.save(model.state_dict(), model_path)
#print(f"✅ Modelo guardado en {model_path}")

# 1️⃣3️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    val_pred = model(X_val).numpy()

# 1️⃣4️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(df["Date"].values[seq_length:train_size+seq_length], df["Close"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df["Date"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
plt.plot(df["Date"].values[train_size+seq_length:], df["Close"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df["Date"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title(f"Predicción de Bitcoin con LSTM (Sin Normalización)")
plt.show()

# 1️⃣5️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()
#%% MODELO DE ENTRENAMIENTO 4: 128 NEURONAS+dropout, GUARDADO EN CARPETA DE ENTRENAMIENTO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Ruta donde guardaremos el modelo
model_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\Kevin\series_temp_entrenar_modelo\lstm_model_256.pth"

# 1️⃣ Generar datos de una onda cosenoidal con patrones de temporada
np.random.seed(4299)
time = np.arange(0, 600, 0.1)  # Onda más larga (4000 puntos)

# 🔄 Onda con estacionalidad: combinación de varias ondas cosenoidales
seasonal_wave = np.cos(time / 5) + 0.5 * np.cos(time / 20) + 0.2 * np.cos(time / 50)
seasonal_wave += 0.002 * time  # Tendencia suave
seasonal_wave += np.random.normal(scale=0.1, size=len(time))  # Ruido normal

# ⚡ Introducir un shock exógeno (evento único de gran impacto)
shock_index = np.random.randint(len(time) // 2, len(time))  # Evento en la segunda mitad
shock_magnitude = np.random.uniform(-11, 11)  # Puede ser positivo o negativo
seasonal_wave[shock_index:] += shock_magnitude  # Aplica el cambio a partir del evento

# 📊 Crear DataFrame final
df = pd.DataFrame({"time_idx": np.arange(len(time)), "value": seasonal_wave})



# 2️⃣ Convertir datos a formato de series temporales
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(df["value"].values, seq_length)

# Dividir datos en entrenamiento y validación
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 3️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 4️⃣ Cargar modelo si existe
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Modelo cargado correctamente desde el disco.")
else:
    print("No se encontró un modelo previo. Se entrenará desde cero.")

# 5️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item()}, Pérdida de validación: {val_loss.item()}")

# 6️⃣ Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# 7️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    val_pred = model(X_val).numpy()

# 8️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], df["value"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df["time_idx"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
plt.plot(df["time_idx"].values[train_size+seq_length:], df["value"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df["time_idx"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Onda Sinusoidal con LSTM")
plt.show()

# 9️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()

#%% MODELO DE ENTRENAMIENTO 4: 128 NEURONAS+dropout en BITCOIN 2013-2021, GUARDADO EN CARPETA DE ENTRENAMIENTO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Ruta donde guardaremos el modelo
model_path = r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\Kevin\series_temp_entrenar_modelo\lstm_model_256.pth"

# 1️⃣ Cargar datos
df_bitcoin = pd.read_csv(r"C:\Users\randy\Desktop\Kevin\STEM\Python\series_temporales\celia_lozano\TimeSeries\data\Bitcoin.csv")

# Convertir la columna 'Date' a tipo datetime
df_bitcoin["Date"] = pd.to_datetime(df_bitcoin["Date"])

# Normalizar la columna "Closing Price (USD)"
scaler = MinMaxScaler(feature_range=(0, 1))
df_bitcoin["Closing Price (USD)"] = scaler.fit_transform(df_bitcoin[["Closing Price (USD)"]])

# 2️⃣ Convertir datos a formato de series temporales
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(df_bitcoin["Closing Price (USD)"].values, seq_length)

# Dividir datos en entrenamiento y validación
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# 3️⃣ Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()

# 4️⃣ Cargar modelo si existe
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Modelo cargado correctamente desde el disco.")
else:
    print("No se encontró un modelo previo. Se entrenará desde cero.")

# 5️⃣ Entrenamiento del modelo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30
training_loss, validation_loss = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Evaluación en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss.item():.6f}, Pérdida de validación: {val_loss.item():.6f}")

# 6️⃣ Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# 7️⃣ Hacer predicciones
model.eval()
with torch.no_grad():
    train_pred = model(X_train).squeeze(-1).numpy()
    val_pred = model(X_val).squeeze(-1).numpy()

# 8️⃣ Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(df_bitcoin["Date"].values[seq_length:train_size+seq_length], df_bitcoin["Closing Price (USD)"].values[seq_length:train_size+seq_length], label="Entrenamiento Real")
plt.plot(df_bitcoin["Date"].values[seq_length:train_size+seq_length], train_pred, label="Entrenamiento Predicho", linestyle="dashed")
plt.plot(df_bitcoin["Date"].values[train_size+seq_length:], df_bitcoin["Closing Price (USD)"].values[train_size+seq_length:], label="Validación Real")
plt.plot(df_bitcoin["Date"].values[train_size+seq_length:], val_pred, label="Validación Predicho", linestyle="dashed")
plt.legend()
plt.title("Predicción de Bitcoin con LSTM")
plt.show()

# 9️⃣ Graficar la evolución de la pérdida
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label="Pérdida de Entrenamiento")
plt.plot(validation_loss, label="Pérdida de Validación")
plt.legend()
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()
#%% MODELO DE ENTRENAMIENTO 5: 128 NEURONAS+dropout en BITCOIN 2021-2025, GUARDADO EN CARPETA DE ENTRENAMIENTO



#%% DATOS SINTETICOS
# 1️⃣ Generar datos de una onda cosenoidal con patrones de temporada
np.random.seed(4299)
time = np.arange(0, 600, 0.1)  # Onda más larga (4000 puntos)

# 🔄 Onda con estacionalidad: combinación de varias ondas cosenoidales
seasonal_wave = np.cos(time / 5) + 0.5 * np.cos(time / 20) + 0.2 * np.cos(time / 50)
seasonal_wave += 0.002 * time  # Tendencia suave
seasonal_wave += np.random.normal(scale=0.1, size=len(time))  # Ruido normal

# ⚡ Introducir un shock exógeno (evento único de gran impacto)
shock_index = np.random.randint(len(time) // 2, len(time))  # Evento en la segunda mitad
shock_magnitude = np.random.uniform(-11, 11)  # Puede ser positivo o negativo
seasonal_wave[shock_index:] += shock_magnitude  # Aplica el cambio a partir del evento

# 📊 Crear DataFrame final
df = pd.DataFrame({"time_idx": np.arange(len(time)), "value": seasonal_wave})







#%% DATOS SINTETICOS
# 1️⃣ Generar datos de una onda cosenoidal con patrones de temporada
np.random.seed(42)
time = np.arange(0, 700, 0.1)  # Onda más larga (4000 puntos)

# 🔄 Onda con estacionalidad: combinación de varias ondas cosenoidales
seasonal_wave = np.cos(time / 5) + 0.5 * np.cos(time / 20) + 0.2 * np.cos(time / 50)
seasonal_wave += 0.002 * time  # Tendencia suave
seasonal_wave += np.random.normal(scale=0.1, size=len(time))  # Ruido normal

# ⚡ Introducir un shock exógeno (evento único de gran impacto)
shock_index = np.random.randint(len(time) // 2, len(time))  # Evento en la segunda mitad
shock_magnitude = np.random.uniform(-11, 11)  # Puede ser positivo o negativo
seasonal_wave[shock_index:] += shock_magnitude  # Aplica el cambio a partir del evento

# 📊 Crear DataFrame final
df = pd.DataFrame({"time_idx": np.arange(len(time)), "value": seasonal_wave})
