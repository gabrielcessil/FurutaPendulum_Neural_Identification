from rotary_pendulum import Rotary_Pendulum
from mechanism_animation import MechanicalSystemAnimation
import numpy as np

# Exemplo de parametros construtivos
params = {
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
rotary_pendulum = Rotary_Pendulum(params)

# Exemplo de função de controle
def u_base(t, q):
    theta_pend_ref = np.pi
    theta_base, theta_pend = q
    T1 = -np.sin((theta_pend_ref - theta_pend) % 2*np.pi) * 2
    return np.array([T1, 0])

# Solução do sistema
initial_state = {"theta_base": 0,
                  "theta_pend": 0,
                  "omega_base": 0,
                  "omega_pend": 0}
tf = 2
t0 = 0
steps = 50

# Resolve a dinamica para essas condicoes
rotary_pendulum.solve_system(control_law=u_base, x0=initial_state, tf=tf, t0=t0, steps=steps)

# Criação da animação
animator = MechanicalSystemAnimation(rotary_pendulum)

# Mostrar animação
animator.create_animation()