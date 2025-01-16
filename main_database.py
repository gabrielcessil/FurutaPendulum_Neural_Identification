import numpy as np
import pandas as pd
from rotary_pendulum import Rotary_Pendulum
from mechanism_animation import MechanicalSystemAnimation
import numpy as np

def create_dataset(N, tf=5, t0=0, steps=100):

    entradas_N = []
    saidas_N = []

    for i in range(N):
      params = {
        "m1": np.random.uniform(low = 0.001, high = 1),  # kg
        "m2": np.random.uniform(low = 0.001, high = 2),  # kg
        "L1": np.random.uniform(low = 0.1, high = 2),  # m
        "L2": np.random.uniform(low = 0.1, high = 2),  # m
        "l1": np.random.uniform(low = 0.1, high = 1.5), # m
        "l2": np.random.uniform(low = 0.1, high = 1.5), # m
        "b1": 0,
        "b2": 0
      }

      params["J1"] = params["m1"]*params["l1"]**2
      params["J2"] = params["m2"]*params["l2"]**2

      pendulum = Rotary_Pendulum(params)

      x0 = {
          "theta_base":0,
          "theta_pen":0,
          "omega_base":0,
          "omega_pen":0,
      }
      
      A = np.random.uniform(low = 5, high = 10)
      a = np.random.uniform(low = 1, high = 50)
      B = np.random.uniform(low = 1, high = 10)
      b = np.random.uniform(low = 1, high = 50)
      C = np.random.uniform(low = 1, high = 10)
      c = np.random.uniform(low = 1, high = 50)
      def u_base(t, q):
          theta_pend_ref = np.pi
          theta_base, theta_pend = q
          T1 = A*np.sin(2*np.pi*t/a) + B*np.sin(2*np.pi*t/b) + C*np.sin(2*np.pi*t/b)
          return np.array([T1, 0])
      
      states = pendulum.solve_system(u_base, x0, tf=tf, t0=t0, steps=steps)
      theta_base, theta_pen = states["theta_base"], states["theta_pen"]
      omega_base, omega_pen = states["omega_base"], states["omega_pen"]

      entradas = np.concatenate([theta_base, theta_pen, omega_base, omega_pen])
      saidas  = np.array(list(params.values()))

      entradas_N.append(entradas)
      saidas_N.append(saidas)
      
      print("Criação de Database: ",i,"/ ",N)

    df_Entradas = pd.DataFrame(entradas_N)
    df_Saidas = pd.DataFrame(saidas_N)

    return df_Entradas, df_Saidas

# CRIAR O DATASET
df_i, df_o = create_dataset(N=1000, tf=2, t0=0, steps=100)
# Salvar os arquivos CSV no diretório temporário
df_i.to_csv("Dataset_Furuta_in.csv", index=False)
df_o.to_csv("Dataset_Furuta_out.csv", index=False)