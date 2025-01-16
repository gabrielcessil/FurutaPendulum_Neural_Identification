import numpy as np
from scipy.integrate import solve_ivp


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Rotary_Pendulum:
  def __init__(self, params):
    self.states = None
    self.params = params
    self.time = None

  def solve_system(self, control_law, x0, tf=10, t0=0, steps=500):
      # Intervalo de tempo
      self.time = np.linspace(t0, tf, steps)
      # Calcula as variaveis de estado para cada instante de tempo
      sol = solve_ivp(self.dynamics, [self.time[0], self.time[-1]], list(x0.values()), t_eval=self.time, args=(control_law,))
      self.states = dict(zip(x0.keys(), sol.y))
      return self.states

  # Dinâmica do sistema
  def dynamics(self, t, x, control_law):
      q = x[:len(x)//2]  # Posições generalizadas (primeiros N//2 elementos)
      q_dot = x[len(x)//2:]  # Velocidades generalizadas (ultimos N//2 elementos)

      M_eval = self.M(q)
      T_eval = control_law(t, q)
      C_eval = self.C(q, q_dot)
      G_eval = self.G(q)

      q_ddot = np.linalg.inv(M_eval) @ (T_eval - C_eval @ q_dot - G_eval)

      return np.concatenate([q_dot, q_ddot])

  # USER INPUTS

  def M(self, q):

      J1, J2 = self.params['J1'], self.params['J2']
      m1, m2 = self.params['m1'], self.params['m2']
      l1, l2 = self.params['l1'], self.params['l2']
      L1, L2 = self.params['L1'], self.params['L2']

      J1_ = J1+m1*l1**2
      J0_ = J1_+m2*L1**2
      J2_ = J2 +m2*l2**2

      M11 = J0_ + J2_*np.sin(q[1])**2
      M12 = m2*L1*l2*np.cos(q[1])
      M21 = m2*L1*l2*np.cos(q[1])
      M22 = J2_
      return np.array([[M11, M12],[M21, M22]])

  def C(self, q, q_dot):

      J1, J2 = self.params['J1'], self.params['J2']
      m1, m2 = self.params['m1'], self.params['m2']
      l1, l2 = self.params['l1'], self.params['l2']
      L1, L2 = self.params['L1'], self.params['L2']
      b1, b2 = self.params['b1'], self.params['b2']

      J2_ = J2 +m2*l2**2

      C11 = b1 + 0.5*q_dot[1]*J2_*np.sin(2*q[1])
      C12 = 0.5*q_dot[1]*J2_*np.sin(2*q[1])-m2*L1*l2*np.sin(q[1])*q_dot[1]
      C21 = -0.5*q_dot[0]*J2_*np.sin(2*q[1])
      C22 = b2

      return np.array([[C11,C12],[C21,C22]])

  def G(self, q):
      # Termos gravitacionais
      m1, m2 = self.params['m1'],self.params['m2']
      L1, L2 = self.params['L1'],self.params['L2']
      g = 9.81
      return np.array([ 0, g*m2*L2*np.sin(q[1])])

  # Calcular as posições no espaço 3D para cada estado (Cinematica direta)
  def direct_kinematics(self, state):
      theta_base, theta_pen = state[0], -state[1]

      L1, L2 = self.params["L1"], self.params["L2"]
      x0, y0, z0 = 0, 0, 0  # Origem fixa

      # Posição do braço base
      x1 = L1 * np.cos(theta_base)
      y1 = L1 * np.sin(theta_base)
      z1 = 0

      # Posição do pêndulo
      x2 = x1 + L2 * np.sin(theta_base) * np.sin(theta_pen)
      y2 = y1 - L2 * np.cos(theta_base) * np.sin(theta_pen)
      z2 = - L2 * np.cos(theta_pen)

      return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)