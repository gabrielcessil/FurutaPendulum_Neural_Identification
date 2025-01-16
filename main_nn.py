import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from datetime import datetime
from torch import isnan 
import sys
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from rotary_pendulum import Rotary_Pendulum
from mechanism_animation import MechanicalSystemAnimation

class MyNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.flatten = nn.Flatten()


        layers = []
        layers.append(nn.Linear(n_in, n_in*2))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n_in*2, n_in*4))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n_in*4, n_in*2))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n_in*2, n_in))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n_in, round(n_in / 2) + 1))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(round(n_in / 2) + 1, round(n_in / 2) + 1))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(round(n_in / 2) + 1, round((round(n_in / 2) + 1 + n_out) / 2)))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(round((round(n_in / 2) + 1 + n_out) / 2), n_out))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n_out, n_out))
        layers.append(nn.Tanh())
        
        # Define a rede
        self.net = nn.Sequential(*layers)
        
        self.output_activation = nn.Softplus()
        

    def forward(self, x):
        y = self.net(x)     
        return self.output_activation(y)
    
def suffle_data(inputs_tensor, labels_tensor):
    
    # Generate a random permutation of sample indices
    permutation = torch.randperm(inputs_tensor.size(0))
    
    # Shuffle samples using the same permutation
    shuffled_inputs = inputs_tensor[permutation, ...]
    shuffled_labels = labels_tensor[permutation, ...]
    
    return shuffled_inputs, shuffled_labels
    
def train_one_epoch(model, train_loader, loss_function, optimizer):
    model.train()

    running_loss = 0.0    
    # Para cada batch
    for batch_inputs, batch_labels in train_loader:
        
        # REALIZA BATCH:
        optimizer.zero_grad() # Reinicia os gradientes para calculo do passo dos pesos
        batch_outputs = model(batch_inputs) # Calcula Saídas
        loss = loss_function(batch_outputs, batch_labels) # Calcula Custo
        loss.backward() # Calcula gradiente em relacao do Custo
        optimizer.step() # Realiza um passo dos pesos
        
        running_loss += loss.item() # Atualizar a perda acumulada da epoca

    # Perda média para a última época
    avg_loss = running_loss / len(train_loader)

    return avg_loss, model

def validate_one_epoch(model, valid_loader, loss_function):
    
    with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
    
      model.eval() # Entre em modo avaliacao, desabilitando funcoes exclusivas de treinamento (ex:Dropout)
      valid_summed_loss = 0.0

      # Para cada sample de validacao
      for valid_input, valid_label in valid_loader:
          # Gera a saida do modelo atual
          valid_output = model(valid_input)
          # Calcula o loss entre output gerado e label
          valid_loss = loss_function(valid_output, valid_label)
          valid_summed_loss += valid_loss.item()
         
      valid_avg_loss = valid_summed_loss / len(valid_loader)
      
      return valid_avg_loss
    

def full_train(model, train_loader, valid_loader, loss_function, optimizer,N_epochs=1000, file_name = "model_weights", include_time=False):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_cost_history = []
    val_cost_history = []
    best_valid_loss = np.inf    
    saved_model = None
    
    for epoch_index in range(N_epochs):
          
      print("\nExecutando epoca ", epoch_index, " /", N_epochs)
      train_avg_loss, model = train_one_epoch( 
          model=model,
          train_loader=train_loader,
          loss_function=loss_function, 
          optimizer=optimizer)
      
      
     
      valid_avg_loss = validate_one_epoch(
          model=model, 
          valid_loader=valid_loader,
          loss_function=loss_function)
    
          
      
      print(' - Average Loss in train {}\n - Average Loss in validation {}'.format(train_avg_loss, valid_avg_loss))
      
      train_cost_history.append(train_avg_loss)
      val_cost_history.append(valid_avg_loss)
      
      
      # Track best performance, and save the model's state
      if valid_avg_loss < best_valid_loss:
          print("Melhor custo encontrado: ", valid_avg_loss, "\n")
          best_valid_loss = valid_avg_loss
          if include_time: model_path = file_name+"_{}_{}.pth".format(timestamp, epoch_index)
          else: model_path = file_name
          torch.save(model.state_dict(), model_path)
      
    return model_path, train_cost_history, val_cost_history


def Plot_Validation(train_cost_history, val_cost_history):
    
    # Trainning Visualization
    x = range(len(train_cost_history))
    plt.plot(x, train_cost_history, label='train', color='blue', marker='none')  # Line 1
    plt.plot(x, val_cost_history, label='validation', color='red', marker='none')     # Line 2
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Two Lines')
    plt.legend()
    plt.show()
    
    train_cost_rate_history = [train_cost_history[i+1]-train_cost_history[i] for i in range(len(train_cost_history)-1)]
    val_cost_rate_history = [val_cost_history[i+1]-val_cost_history[i] for i in range(len(val_cost_history)-1)]
    x = range(len(val_cost_rate_history))
    plt.plot(x, train_cost_rate_history, label='train', color='blue', marker='none')  # Line 1
    plt.plot(x, val_cost_rate_history, label='validation', color='red', marker='none')     # Line 2
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Two Lines')
    plt.legend()
    plt.show()
    
def Test_Case(
        model,
        u_base,
        test_params,
        scaler, 
        initial_state = {"theta_base": 0,
                          "theta_pend": 0,
                          "omega_base": 0,
                          "omega_pend": 0},
        tf = 2,
        t0 = 0,
        steps = 100
        ):
    


    # Simulando Sistema de Pendulo
    test_rotary_pendulum = Rotary_Pendulum(test_params)
    test_rotary_pendulum.solve_system(control_law=u_base, x0=initial_state, tf=tf, t0=t0, steps=steps)
    # Usando simulacao para criar entradas da rede
    theta_pen, omega_pen = test_rotary_pendulum.states["theta_pend"], test_rotary_pendulum.states["omega_pend"]
    theta_base, omega_base = test_rotary_pendulum.states["theta_base"], test_rotary_pendulum.states["omega_base"]
    entradas = np.concatenate([theta_base, theta_pen, omega_base, omega_pen])
    entradas = scaler.transform(entradas.reshape(1, -1))
    entradas = torch.tensor(entradas, dtype=torch.float32)
    
    # Usando entradas para prever parametros estruturais conforme modelo salvo
    model.eval()
    pred_params = model(entradas).tolist()[0]
    
    result = {}
    for param, param_name in zip(pred_params, test_params.keys()): result[param_name] = param
        
    return result
    
#### LOADING DATA
df_i = pd.read_csv("Dataset_Furuta_in.csv")
df_o = pd.read_csv("Dataset_Furuta_out.csv")
n_in = df_i.shape[1]
n_out = df_o.shape[1]
print('Dataframe de entradas: {} atributos de {} amostras '.format(df_i.shape[1], df_i.shape[0]))
print('Dataframe de saídas: {} atributos de {} amostras '.format(df_o.shape[1], df_o.shape[0]))

#### HYPER PARAMETERS
learning_rate = 0.01 
test_fraction = 0.1 # Fraction of data used for testing rather then training
N_epochs = 8000 # Number of iterations over all data 
batch_size = 250  # Number of samples per batch


#### PRE-PROCESSING DATA
# Separar em treino, teste, validacao
df_i_train, df_i_valid, df_o_train, df_o_valid = train_test_split(df_i, df_o, test_size=test_fraction, random_state=42, shuffle=True)
# Normalizar as entradas em relação ao Max/Min do treino
scaler = MinMaxScaler()
scaler.fit(df_i_train.to_numpy())
# Normaliza treino e teste baseado na populacao dos dados de treino
train_inputs = scaler.transform(df_i_train.to_numpy())
valid_inputs = scaler.transform(df_i_valid.to_numpy())
# Converter os arrays em tensores PyTorch
train_inputs = torch.tensor(df_i_train.to_numpy(), dtype=torch.float32)
train_labels = torch.tensor(df_o_train.to_numpy(), dtype=torch.float32)
valid_inputs = torch.tensor(df_i_valid.to_numpy(), dtype=torch.float32)
valid_labels = torch.tensor(df_o_valid.to_numpy(), dtype=torch.float32)



# Step 1: Combine tensors into a dataset
valid_loader = DataLoader(TensorDataset(valid_inputs, valid_labels), batch_size=batch_size, shuffle=True)
train_loader = DataLoader(TensorDataset(train_inputs, train_labels), batch_size=batch_size, shuffle=True)



#### MODEL DEFINITION
model = MyNet(n_in, n_out)
loss_function = nn.MSELoss() #nn.MSELoss() nn.L1Loss, nn.HuberLoss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#### MODEL TRAINNING
model_path, train_cost_history, val_cost_history = full_train(
                                                    model,
                                                    train_loader,
                                                    valid_loader,
                                                    loss_function,
                                                    optimizer,
                                                    N_epochs=N_epochs)

Plot_Validation(train_cost_history, val_cost_history)

#### MODEL TESTING
def u_base(t, q):
    theta_pend_ref = np.pi
    theta_base, theta_pend = q
    T1 = -np.sin((theta_pend_ref - theta_pend) % 2*np.pi) * 2
    return np.array([T1, 0])

test_params = {
    "m1": 0.2,  # kg
    "m2": 0.7,  # kg
    "L1": 1.4,  # m
    "L2": 0.5,  # m
    "l1": 1.4, # m
    "l2": 0.5, # m
    "J1": 0.2*1.4**2,
    "J2": 0.7*0.5**2,
    "b1": 0,
    "b2": 0
    }

final_model = MyNet(n_in, n_out)
final_model.load_state_dict(torch.load(model_path, weights_only=True)) # Safe way of Loading models (Recreate model from states only)
pred_params = Test_Case(final_model, u_base, test_params, scaler)

pred_rotary_pendulum = Rotary_Pendulum(pred_params)

# Solução do sistema
initial_state = {"theta_base": 0,
                  "theta_pend": 0,
                  "omega_base": 0,
                  "omega_pend": 0}
tf = 2
t0 = 0
steps = 50

# Resolve a dinamica para essas condicoes
pred_rotary_pendulum.solve_system(control_law=u_base, x0=initial_state, tf=tf, t0=t0, steps=steps)

# Criação da animação
animator = MechanicalSystemAnimation(pred_rotary_pendulum)
animator.create_animation("Pred_Pendulum")








#### FUNCAO CUSTO BASEADA NA SERIE TEMPORAL
"""
def u_base(t, q):
    theta_pend_ref = np.pi
    theta_base, theta_pend = q
    T1 = -np.sin((theta_pend_ref - theta_pend) % 2*np.pi) * 2
    return np.array([T1, 0])

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        # Certifique-se de que output e target são 1D tensors
        output = output.view(-1)
        target = target.view(-1)

        # Cria os dicionários dos parâmetros previstos e reais
        pred_params = {
            "m1": output[0].item(),  
            "m2": output[1].item(),  
            "L1": output[2].item(),  
            "L2": output[3].item(),
            "l1": output[4].item(), 
            "l2": output[5].item(), 
            "J1": output[6].item(),
            "J2": output[7].item(),
            "b1": output[8].item(),
            "b2": output[9].item()
        }
        
        real_params = {
            "m1": target[0].item(),  
            "m2": target[1].item(),  
            "L1": target[2].item(),
            "L2": target[3].item(),  
            "l1": target[4].item(),
            "l2": target[5].item(),  
            "J1": target[6].item(),
            "J2": target[7].item(),
            "b1": target[8].item(),
            "b2": target[9].item()
        }

        initial_state = {"theta_base": 0,
                         "theta_pend": 0,
                         "omega_base": 0,
                         "omega_pend": 0}
        tf = 2
        t0 = 0
        steps = 100

        # Simula com parâmetros previstos
        test_rotary_pendulum = Rotary_Pendulum(pred_params)
        test_rotary_pendulum.solve_system(control_law=u_base, x0=initial_state, tf=tf, t0=t0, steps=steps)
        
        # Obtém os estados previstos diretamente como tensores PyTorch
        theta_pen = torch.tensor(test_rotary_pendulum.states["theta_pend"], dtype=torch.float32, requires_grad=True)
        omega_pen = torch.tensor(test_rotary_pendulum.states["omega_pend"], dtype=torch.float32, requires_grad=True)
        theta_base = torch.tensor(test_rotary_pendulum.states["theta_base"], dtype=torch.float32, requires_grad=True)
        omega_base = torch.tensor(test_rotary_pendulum.states["omega_base"], dtype=torch.float32, requires_grad=True)
        
        pred_signals = torch.cat([theta_base, theta_pen, omega_base, omega_pen])

        # Simula com parâmetros reais
        test_rotary_pendulum = Rotary_Pendulum(real_params)
        test_rotary_pendulum.solve_system(control_law=u_base, x0=initial_state, tf=tf, t0=t0, steps=steps)
        
        # Obtém os estados reais diretamente como tensores PyTorch
        theta_pen = torch.tensor(test_rotary_pendulum.states["theta_pend"], dtype=torch.float32, requires_grad=True)
        omega_pen = torch.tensor(test_rotary_pendulum.states["omega_pend"], dtype=torch.float32, requires_grad=True)
        theta_base = torch.tensor(test_rotary_pendulum.states["theta_base"], dtype=torch.float32, requires_grad=True)
        omega_base = torch.tensor(test_rotary_pendulum.states["omega_base"], dtype=torch.float32, requires_grad=True)
        
        real_signals = torch.cat([theta_base, theta_pen, omega_base, omega_pen])

        # Calcula o erro quadrático médio
        mse = nn.MSELoss()
        return mse(pred_signals, real_signals)


loss_fn = CustomLoss()
model = MyNet(n_in, n_out)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model_path, train_cost_history, val_cost_history = full_train(
                                                    model,
                                                    train_loader,
                                                    valid_loader,
                                                    loss_fn,
                                                    optimizer,
                                                    N_epochs=N_epochs)

Plot_Validation(train_cost_history, val_cost_history)

test_params = {
    "m1": 0.2,  # kg
    "m2": 0.7,  # kg
    "L1": 1.4,  # m
    "L2": 0.5,  # m
    "l1": 1.4, # m
    "l2": 0.5, # m
    "J1": 0.2*1.4**2,
    "J2": 0.7*0.5**2,
    "b1": 0,
    "b2": 0
    }

final_model = MyNet(n_in, n_out)
final_model.load_state_dict(torch.load(model_path, weights_only=True)) # Safe way of Loading models (Recreate model from states only)
pred_params = Test_Case(final_model, u_base, test_params, scaler)
"""