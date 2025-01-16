import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

class MechanicalSystemAnimation:
    def __init__(self, mechanism, system_name="Sistema Mecânico"):
        self.system_name = system_name
        self.time = mechanism.time
        self.states = mechanism.states
        self.xyz_t = [mechanism.direct_kinematics([state[i] for state in self.states.values()]) for i in range(len(self.time))]

    def setup_figure(self):
        """Configura a figura para animação."""
        return make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
            subplot_titles=(f"{self.system_name} in 3D", "States")
        )

    def create_frames(self):
        """Cria os frames para a animação."""
        frames = []
        # Cria um frame para cada instante de tempo
        for i in range(len(self.time)):
            frames.append(
              go.Frame(
                  data = [self.create_3D_plot(i)] + self.create_2D_plot(i)
              )
            )
        return frames

    def create_3D_plot(self, i):
      positions_i = self.xyz_t[i]
      x, y, z = [], [], []

      for point in positions_i:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

      # Scatter 3D plot: posicoes cartesianas no instante i para cada ponto
      return go.Scatter3d(
              x=x,
              y=y,
              z=z,
              mode='lines+markers',
              line=dict(color=['cyan', 'black'], width=5),
              marker=dict(size=8, color=['blue', 'purple', 'magenta'])
            )

    def create_2D_plot(self, i):
      # Lista de estados, considerando ate o instante i, para plot 2D
      n_variables = len(self.states.values())

      colorscale = 'Viridis'  # continuous color scale from Plotly
      colors = px.colors.sample_colorscale(colorscale, n_variables)

      result_plots = []
      for state_color, (state_name,state) in zip(colors,self.states.items()):
          result_plots.append(
              go.Scatter(
                  x=self.time[:i],
                  y=state[:i],
                  mode="lines",
                  line=dict(color=state_color),
                  name=state_name
              )
          )

      return result_plots

    def create_initial_plot(self):
        """Cria o gráfico inicial para a animação (t0)."""
        x,y,z = self.xyz_t[0][0], self.xyz_t[0][1], self.xyz_t[0][2]
        return go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(size=5)
        )

    def create_animation(self, file_name = "Mechanism_Animation"):
        """Cria e exibe a animação do sistema mecânico."""
        frames = self.create_frames()

        fig = self.setup_figure()

        # Adiciona o gráfico 3D inicial
        fig.add_trace(self.create_initial_plot(), row=1, col=1)
        # Adiciona os gráficos 2D iniciais
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", line=dict(color="green")), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", line=dict(color="orange")), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", line=dict(color="red")), row=1, col=2)

        # Adiciona os frames
        fig.frames = frames

        # Configuração de animação
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )],
            scene=dict(
                xaxis=dict(range=[-2, 2]),
                yaxis=dict(range=[-2, 2]),
                zaxis=dict(range=[-2, 2]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1)
            ),
            xaxis2=dict(title="Tempo (s)"),
            yaxis2=dict(title="Velocidade (m/s)")
        )

        fig.show()
        
        # Save the animation as an HTML file
        pio.write_html(fig, file=file_name+".html", auto_open=True)
        print(f"Animation saved as {file_name}")
