import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

# Configure FFmpeg path
mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\diana.dospinescu\Desktop\Diana\code\temp, pres, umid\ffmpeg-2025-03-27-git-114fccc4a5-full_build\bin\ffmpeg.exe"


ROOM_DIM = (5.0, 5.0)     # meters
DX = 0.1                  # spatial step
DY = DX

# Grid
x = np.arange(0, ROOM_DIM[0] + DX, DX)
y = np.arange(0, ROOM_DIM[1] + DY, DY)
X, Y = np.meshgrid(x, y)

NY, NX = X.shape

# Source position (corner)
SOURCE_I = 0
SOURCE_J = 0

# Diffusivities converted from cm^2/s to m^2/s
DIFFUSIVITY = {
    'temp': 0.2e-4,
    'pressure': 0.5e-4,
    'humidity': 0.25e-4
}

AMBIENT = {
    'temp': 20.0,
    'pressure': 1013.25,
    'humidity': 30.0
}

SOURCE = {
    'temp': 30.0,
    'pressure': 1010.0,
    'humidity': 65.0
}


D_max = max(DIFFUSIVITY.values())
DT_stable = DX**2 / (4 * D_max)
DT = 0.8 * DT_stable   # keep some safety margin

SIM_TIME = 7200        # seconds
NUM_STEPS = int(SIM_TIME / DT)

# For animation
NUM_FRAMES = 60
FRAME_INTERVAL = 50
frame_indices = np.linspace(0, NUM_STEPS - 1, NUM_FRAMES, dtype=int)

# -----------------------------
# Initial conditions
# -----------------------------
temp = np.full((NY, NX), AMBIENT['temp'], dtype=float)
pressure = np.full((NY, NX), AMBIENT['pressure'], dtype=float)
humidity = np.full((NY, NX), AMBIENT['humidity'], dtype=float)

# Fixed source in the corner
temp[SOURCE_I, SOURCE_J] = SOURCE['temp']
pressure[SOURCE_I, SOURCE_J] = SOURCE['pressure']
humidity[SOURCE_I, SOURCE_J] = SOURCE['humidity']

def apply_boundary(u):
    """
    Simple Neumann-like boundary handling by copying adjacent values.
    Keeps the field numerically well-behaved near edges.
    """
    u[0, 1:] = u[1, 1:]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    return u

def enforce_source(u, source_value):
    # Keep a small source patch so a boundary source can couple to interior nodes.
    for di in (0, 1):
        for dj in (0, 1):
            ii = SOURCE_I + di
            jj = SOURCE_J + dj
            if 0 <= ii < NY and 0 <= jj < NX:
                u[ii, jj] = source_value
    return u

def euler_diffusion_step(u, D, dt, dx, dy):
    u_new = u.copy()

    # 2D Laplacian on interior nodes
    laplacian = (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )

    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * D * laplacian
    return u_new


saved_temp = []
saved_pressure = []
saved_humidity = []
saved_times = []

for step in range(NUM_STEPS):
    temp = euler_diffusion_step(temp, DIFFUSIVITY['temp'], DT, DX, DY)
    pressure = euler_diffusion_step(pressure, DIFFUSIVITY['pressure'], DT, DX, DY)
    humidity = euler_diffusion_step(humidity, DIFFUSIVITY['humidity'], DT, DX, DY)

    temp = apply_boundary(temp)
    pressure = apply_boundary(pressure)
    humidity = apply_boundary(humidity)

    temp = enforce_source(temp, SOURCE['temp'])
    pressure = enforce_source(pressure, SOURCE['pressure'])
    humidity = enforce_source(humidity, SOURCE['humidity'])

    if step in frame_indices:
        saved_temp.append(temp.copy())
        saved_pressure.append(pressure.copy())
        saved_humidity.append(humidity.copy())
        saved_times.append(step * DT)


fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Temperature (°C)', 'Pressure (hPa)', 'Humidity (%)']
cmaps = ['hot', 'viridis', 'Blues']

clim = {
    'temp': (AMBIENT['temp'], SOURCE['temp']),
    'pressure': (min(AMBIENT['pressure'], SOURCE['pressure']),
                 max(AMBIENT['pressure'], SOURCE['pressure'])),
    'humidity': (AMBIENT['humidity'], SOURCE['humidity'])
}

initial_fields = [saved_temp[0], saved_pressure[0], saved_humidity[0]]
keys = ['temp', 'pressure', 'humidity']
imgs = []

for ax, field, title, cmap, key in zip(axs, initial_fields, titles, cmaps, keys):
    im = ax.imshow(
        field,
        extent=[0, ROOM_DIM[0], 0, ROOM_DIM[1]],
        origin='lower',
        cmap=cmap,
        vmin=clim[key][0],
        vmax=clim[key][1],
        animated=True
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax)
    imgs.append(im)

time_text = fig.text(0.5, 0.98, '', ha='center', va='top')

def animate(frame):
    imgs[0].set_data(saved_temp[frame])
    imgs[1].set_data(saved_pressure[frame])
    imgs[2].set_data(saved_humidity[frame])

    total_seconds = int(round(saved_times[frame]))
    minutes, seconds = divmod(total_seconds, 60)
    time_text.set_text(f'Time: {minutes} min {seconds} sec')
    return imgs + [time_text]

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(saved_temp),
    interval=FRAME_INTERVAL,
    blit=False
)

ani.save("diffusion_euler_animation.mp4", writer="ffmpeg", dpi=100)
plt.show()