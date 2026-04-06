import matplotlib.pyplot as plt
import scipy.special as sp
import numpy as np
import matplotlib.animation as animation 
import matplotlib as mpl
import math
import pandas as pd 
import random

# Configure FFmpeg path
mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\diana.dospinescu\Desktop\Diana\code\temp, pres, umid\ffmpeg-2025-03-27-git-114fccc4a5-full_build\bin\ffmpeg.exe"

# Physical parameters
ROOM_DIM = (5.0, 5.0)  # (length, width) in meters
RESOLUTION = 0.1        # Grid spacing in meters
SOURCE_POS = (0.0, 0.0) # Corner source at room origin

# Diffusion parameters (cm²/s)
DIFFUSIVITY = {
    'temp': 0.2,    # Thermal diffusivity in cm²/s
    'humidity': 0.25, # Water vapor diffusivity in cm²/s
    'pressure': 0.5  # Air pressure diffusivity in cm²/s 
}

# Ambient conditions
AMBIENT = {
    'temp': 20.0,      # °C
    'pressure': 1013.25, # hPa
    'humidity': 30    # Fraction (30%)
}

# Source conditions
SOURCE = {
    'temp': 30.0,      # °C
    'pressure': 1010.0, # hPa
    'humidity': 65    # Fraction (65%)
}

# Time parameters
SIM_TIME = 7200 # Total simulation time in seconds (4 hours)
NUM_FRAMES = 60
FRAME_INTERVAL = 50  # ms between animation frames
times = np.linspace(0, SIM_TIME, NUM_FRAMES)  # Time points for animation

# Create grid
x = np.arange(0, ROOM_DIM[0] + RESOLUTION, RESOLUTION)
y = np.arange(0, ROOM_DIM[1] + RESOLUTION, RESOLUTION)
X, Y = np.meshgrid(x, y)
d = np.sqrt((X - SOURCE_POS[0])**2 + (Y - SOURCE_POS[1])**2)

def calculate_diffusion(D, Tx, Px, Hx, t):
    """Return results and individual matrices for given parameters at time t"""
    # Convert diffusivities from cm²/s to m²/s
    alpha = DIFFUSIVITY['temp'] * 1e-4
    dp = DIFFUSIVITY['pressure'] * 1e-4
    dh = DIFFUSIVITY['humidity'] * 1e-4

    # Avoid division by zero at t=0 while keeping the expected initial profile.
    t = max(t, 1e-12)

    results = {}
    results['temp'] = AMBIENT['temp'] + (Tx - AMBIENT['temp']) * (1 - sp.erf(D / np.sqrt(4 * alpha * t)))
    results['pressure'] = AMBIENT['pressure'] + (Px - AMBIENT['pressure']) * (1 - sp.erf(D / np.sqrt(4 * dp * t)))
    results['humidity'] = AMBIENT['humidity'] + (Hx - AMBIENT['humidity']) * (1 - sp.erf(D / np.sqrt(4 * dh * t)))
    
    return results, results['temp'], results['pressure'], results['humidity']

# Set up figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
imgs = []

# Initial plot at first animation time
results, temp_matrix, pressure_matrix, humidity_matrix = calculate_diffusion(
    d, SOURCE['temp'], SOURCE['pressure'], SOURCE['humidity'], times[0]
)
titles = ['Temperature (°C)', 'Pressure (hPa)', 'Humidity (%)']
cmaps = ['hot', 'viridis', 'Blues']

# Set consistent color limits
clim = {
    'temp': (AMBIENT['temp'], SOURCE['temp']),
    'pressure': (min(AMBIENT['pressure'], SOURCE['pressure']),
                 max(AMBIENT['pressure'], SOURCE['pressure'])),
    'humidity': (AMBIENT['humidity'], SOURCE['humidity'])
}

for ax, title, cmap, key in zip(axs, titles, cmaps, results.keys()):
    im = ax.imshow(results[key], 
                  extent=[0, ROOM_DIM[0], 0, ROOM_DIM[1]],
                  origin='lower', 
                  cmap=cmap,
                  animated=True,
                  vmin=clim[key][0], vmax=clim[key][1])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    imgs.append(im)

time_text = fig.text(0.5, 0.98, '', ha='center', va='top')

# Animation function
def animate(frame):
    current_time = times[frame]
    results, _, _, _ = calculate_diffusion(d, SOURCE['temp'], SOURCE['pressure'], SOURCE['humidity'], current_time)
    
    for im, key in zip(imgs, results.keys()):
        im.set_data(results[key])

    total_seconds = int(round(current_time))
    minutes, seconds = divmod(total_seconds, 60)
    time_text.set_text(f'Time: {minutes} min {seconds} sec')
    return imgs + [time_text]

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, 
                              interval=FRAME_INTERVAL, blit=False)

# Save animation
ani.save('diffusion_animation.mp4', writer='ffmpeg', dpi=100)

plt.show()