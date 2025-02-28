import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#primero vamos a generar datos inventados
def generate_data(seq_lenght, num_samples):
    x = np.linspace(0, 50, num_samples) #con esto creamos DATOS SECUENCIALES
    y = np.sin(x) #como ejemplo de prediccion de puntos usaremos una funcion sinusoidal
    data = [y[i:i + seq_lenght + 1] for i in range (len(y) - seq_lenght)] #aqui obtengo las subsecuencias
    data = np.array(data)
    return data[:,:-1], data[:,-1] #aqui cojo las caracteristicas y el objetivo

seq_lenght = 10
num_samples = 200
X,y = generate_data(seq_lenght, num_samples) #de aqui vamos a sacar nuestra caracteristicas y objetivos 

#CONVERTIR A TENSORES
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) #cambiamos a tensores porque son la estructura basica si trabajamos con pytorch
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) #inserto una nueva dimension al final para cumplir con la estructura

#ENTRENAMIENTO Y VALIDACION
train_size = int(0.8 * len(X_tensor)) #cojo solo el 80% para el tamaño de entrenamiento
X_train, X_val = X_tensor[:train_size], X_tensor[train_size:] #con esto tenemos el 80% para entramiento y 20%validacion para x para X
y_train, y_val = y_tensor[:train_size], y_tensor[train_size:] #lo mismo ahora para los objetivos y

#DEFINIMOS EL MODELO DE RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN,self).__init__() #para comprobar que va como queremos
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) #CAPA RECURRENTE
        self.fc = nn.Linear(hidden_size, output_size) #CAPA TOTALMENTE CONECTADA
        
    def forward(self, x):
        out, _ = self.rnn(x) #pasara a traves de toda la rnn
        out = self.fc(out[:,-1,:])
        return out 
    
#INICIALIZAR EL MODELO, FUNCION DE PERDIDA Y OPTIMIZADOR
input_size = 1 #recibimos una dimension
hidden_size = 16 #ESTE ES EL NUMERO DE NEURONAS EN LA CAPA OCULTA
output_size = 1 #nos da de salida una dimension

modelo = SimpleRNN(input_size, hidden_size, output_size) #aqui estamos llamando a la clase con nuestro modelo

criterion = nn.MSELoss() #MSE
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01) #AQUI AJUSTAMOS EL LEARNING RATE
#este es el lr que menos problemas da, despues podemos modificarlo a nuestro gusto
#adam es uno de tantos optimizadores pero en general es el que menos problemas y mas flexible puede ser

#ENTRENAMIENTO DEL MODELO
num_epochs = 100 
for epoch in range(num_epochs):
    modelo.train() #aqui cambiamos a modo entrenamiento
    outputs = modelo(X_train) #aqui estamos realizando el calculo del entrenamiento, son los valores predichos en entrenamiento
    loss = criterion(outputs, y_train) #aqui hacemos el MSE entre lo calculado en el entrenamiento vs el real
    
    optimizer.zero_grad() #calculamos que no se arrastre el gradiente calculado
    loss.backward() #backpropagation: calculo del gradiente de la funcion perdida
    optimizer.step() #despues de encontrar los gradientes con los que reducimos el erros los utilizamos, es un paso de optimizacion para elegir eso
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        
#EVALUACION DEL MODELO

modelo.eval() #cambio a modo de evaluacion
with torch.no_grad(): #para no hacerlo tan complejo decido
    predictions = modelo(X_val)
    loss_val = criterion(predictions, y_val)
    print(f"Validation Loss: {loss_val.item():.4f}")
    
#VISUALIZAR LOS DATOS
plt.figure(figsize=(12,6))
plt.plot(y_val.numpy(), label ='real')
plt.plot(predictions.numpy(), label='predicciones')
plt.legend()
plt.title("Predicción de series temporales")
plt.show()


    