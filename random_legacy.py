import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Configure FFmpeg path (same setup as diffusion.py)
mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\diana.dospinescu\Desktop\Diana\code\temp, pres, umid\ffmpeg-2025-03-27-git-114fccc4a5-full_build\bin\ffmpeg.exe"

# 2D room setup for animation
ROOM_DIM = (5.0, 5.0)
RESOLUTION = 0.1
SOURCE_POS = (0.0, 0.0)

# Time setup for animation
SIM_TIME = 7200
NUM_FRAMES = 60
FRAME_INTERVAL = 50
times = np.linspace(0, SIM_TIME, NUM_FRAMES)

# Load the dataset
df = pd.read_csv('room_diffusion_2D.csv')
df = df.dropna()

# Inputs and targets
X = df[['Tx', 'Px', 'Hx', 'D']]
y_T = df['Ty']
y_P = df['Py']
y_H = df['Hy']

# One single split for all targets
X_train, X_test, yT_train, yT_test, yP_train, yP_test, yH_train, yH_test = train_test_split(
    X, y_T, y_P, y_H, test_size=0.2, random_state=42
)

# Temperature model
rf_T = RandomForestRegressor(
    n_estimators=4,
    max_depth=4,
    random_state=42
)
rf_T.fit(X_train, yT_train)
yT_pred = rf_T.predict(X_test)

# Pressure model
rf_P = RandomForestRegressor(
    n_estimators=4,
    max_depth=4,
    random_state=42
)
rf_P.fit(X_train, yP_train)
yP_pred = rf_P.predict(X_test)

# Humidity model
rf_H = RandomForestRegressor(
    n_estimators=4,
    max_depth=4,
    random_state=42
)
rf_H.fit(X_train, yH_train)
yH_pred = rf_H.predict(X_test)

def evaluate_regression(y_true, y_pred, name="target"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{name}:")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  R2   = {r2:.6f}")
    print()

print("Regression Metrics:")
evaluate_regression(yT_test, yT_pred, "Ty")
evaluate_regression(yP_test, yP_pred, "Py")
evaluate_regression(yH_test, yH_pred, "Hy")

comparison_T = pd.DataFrame({
    "Ty_true": yT_test.values,
    "Ty_pred": yT_pred
})
comparison_T["error"] = comparison_T["Ty_pred"] - comparison_T["Ty_true"]

comparison_P = pd.DataFrame({
    "Py_true": yP_test.values,
    "Py_pred": yP_pred
})
comparison_P["error"] = comparison_P["Py_pred"] - comparison_P["Py_true"]

comparison_H = pd.DataFrame({
    "Hy_true": yH_test.values,
    "Hy_pred": yH_pred
})
comparison_H["error"] = comparison_H["Hy_pred"] - comparison_H["Hy_true"]

print(comparison_T.head(10))
print(comparison_P.head(10))
print(comparison_H.head(10))

results = X_test.copy()
results["Ty_true"] = yT_test.values
results["Ty_pred"] = yT_pred
results["Py_true"] = yP_test.values
results["Py_pred"] = yP_pred
results["Hy_true"] = yH_test.values
results["Hy_pred"] = yH_pred

results.to_csv("rf_predictions_results.csv", index=False)
print("Saved to rf_predictions_results.csv")

source_tx = float(df['Tx'].iloc[0])
source_px = float(df['Px'].iloc[0])
source_hx = float(df['Hx'].iloc[0])

ambient = {
    'temp': float(df['Ty'].min()),
    'pressure': float(df['Py'].min()),
    'humidity': float(df['Hy'].min())
}

x = np.arange(0, ROOM_DIM[0] + RESOLUTION, RESOLUTION)
y = np.arange(0, ROOM_DIM[1] + RESOLUTION, RESOLUTION)
Xg, Yg = np.meshgrid(x, y)
d = np.sqrt((Xg - SOURCE_POS[0]) ** 2 + (Yg - SOURCE_POS[1]) ** 2)

def predict_steady_state(distance_matrix):
    flat_d = distance_matrix.ravel()
    features = pd.DataFrame({
        'Tx': np.full(flat_d.shape, source_tx),
        'Px': np.full(flat_d.shape, source_px),
        'Hx': np.full(flat_d.shape, source_hx),
        'D': flat_d
    })
    temp_pred = rf_T.predict(features).reshape(distance_matrix.shape)
    pressure_pred = rf_P.predict(features).reshape(distance_matrix.shape)
    humidity_pred = rf_H.predict(features).reshape(distance_matrix.shape)
    return {
        'temp': temp_pred,
        'pressure': pressure_pred,
        'humidity': humidity_pred
    }

steady_results = predict_steady_state(d)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
imgs = []
titles = ['Temperature (°C)', 'Pressure (hPa)', 'Humidity (%)']
cmaps = ['hot', 'viridis', 'Blues']

clim = {
    'temp': (min(ambient['temp'], source_tx), max(ambient['temp'], source_tx)),
    'pressure': (min(ambient['pressure'], source_px), max(ambient['pressure'], source_px)),
    'humidity': (min(ambient['humidity'], source_hx), max(ambient['humidity'], source_hx))
}

initial = {
    'temp': np.full_like(d, ambient['temp']),
    'pressure': np.full_like(d, ambient['pressure']),
    'humidity': np.full_like(d, ambient['humidity'])
}

for ax, title, cmap, key in zip(axs, titles, cmaps, ['temp', 'pressure', 'humidity']):
    im = ax.imshow(
        initial[key],
        extent=[0, ROOM_DIM[0], 0, ROOM_DIM[1]],
        origin='lower',
        cmap=cmap,
        animated=True,
        vmin=clim[key][0],
        vmax=clim[key][1]
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    imgs.append(im)

time_text = fig.text(0.5, 0.98, '', ha='center', va='top')

def animate(frame):
    progress = frame / max(NUM_FRAMES - 1, 1)
    front_radius = progress * d.max()
    mask = d <= front_radius

    current = {
        'temp': np.full_like(d, ambient['temp']),
        'pressure': np.full_like(d, ambient['pressure']),
        'humidity': np.full_like(d, ambient['humidity'])
    }

    for key in current:
        current[key][mask] = steady_results[key][mask]

    for im, key in zip(imgs, ['temp', 'pressure', 'humidity']):
        im.set_data(current[key])

    total_seconds = int(round(times[frame]))
    minutes, seconds = divmod(total_seconds, 60)
    time_text.set_text(f'Time: {minutes} min {seconds} sec')

    return imgs + [time_text]

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=NUM_FRAMES,
    interval=FRAME_INTERVAL,
    blit=False
)

ani.save('random_forest_diffusion_animation.mp4', writer='ffmpeg', dpi=100)
print('Saved to random_forest_diffusion_animation.mp4')

plt.show()