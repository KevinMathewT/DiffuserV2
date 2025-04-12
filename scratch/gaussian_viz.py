import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

mu = np.array([0, 0])
cov = np.array([[3.0, -2.4], [-2.4, 3.0]])
data = np.random.multivariate_normal(mu, cov, size=10000)

hb = ax.hexbin(data[:,0], data[:,1], gridsize=50, cmap='viridis')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("2D Gaussian with Covariance Matrix")
ax.set_xlabel("x")
ax.set_ylabel("y")

ax_sigma11 = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_sigma12 = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_sigma21 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_sigma22 = plt.axes([0.25, 0.10, 0.65, 0.03])

s11 = Slider(ax_sigma11, 'σ11', 0.1, 5.0, valinit=3.0)
s12 = Slider(ax_sigma12, 'σ12', -4.0, 4.0, valinit=-2.4)
s21 = Slider(ax_sigma21, 'σ21', -4.0, 4.0, valinit=-2.4)
s22 = Slider(ax_sigma22, 'σ22', 0.1, 5.0, valinit=3.0)

def update(val):
    c = np.array([[s11.val, s12.val], [s21.val, s22.val]])
    if np.all(np.linalg.eigvals(c) > 0):
        ax.clear()
        d = np.random.multivariate_normal(mu, c, size=10000)
        ax.hexbin(d[:,0], d[:,1], gridsize=50, cmap='viridis')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"σ11={s11.val:+.2f}, σ12={s12.val:+.2f}\nσ21={s21.val:+.2f}, σ22={s22.val:+.2f}")
        fig.canvas.draw_idle()

s11.on_changed(update)
s12.on_changed(update)
s21.on_changed(update)
s22.on_changed(update)

plt.show()