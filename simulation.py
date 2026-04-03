"""
Симуляция Творения: предсказуемость эволюции Вселенной
Автор: Абдунодир (Ташкент)
Лицензия: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ========== ПАРАМЕТРЫ СИМУЛЯЦИИ ==========
NUM_PARTICLES = 120
BOX_SIZE = 180
H0 = 0.02
G = 1.2
DARK_STRENGTH = 0.0008
DARK_SCALE = 100.0
NOISE_STRENGTH = 0.5
DT = 0.05
STEPS = 400
PREDICT_STEPS = 60
np.random.seed(42)

# ========== СОЗДАНИЕ ЧАСТИЦ ==========
masses = np.random.uniform(0.5, 2.0, NUM_PARTICLES)
pos0 = np.random.uniform(-BOX_SIZE/2, BOX_SIZE/2, (NUM_PARTICLES, 2))
r0 = np.sqrt(pos0[:,0]**2 + pos0[:,1]**2)
vel0 = np.zeros_like(pos0)
for i in range(NUM_PARTICLES):
    if r0[i] > 0:
        vel0[i] = H0 * pos0[i]
    vel0[i] += np.random.uniform(-0.1, 0.1, 2)

# ========== ФИЗИЧЕСКИЕ ФУНКЦИИ ==========
def accelerations(pos, masses, G, dark_strength, dark_scale):
    n = len(masses)
    a = np.zeros_like(pos)
    for i in range(n):
        for j in range(i+1, n):
            dr = pos[j] - pos[i]
            dist = np.linalg.norm(dr) + 0.5
            f_grav = G * masses[i] * masses[j] / dist**2
            f_dark = dark_strength * masses[i] * masses[j] * (dist / dark_scale)
            f_total = -f_grav + f_dark
            a[i] += f_total * dr / dist / masses[i]
            a[j] -= f_total * dr / dist / masses[j]
    return a

def evolve(pos0, vel0, masses, G, dark_strength, dark_scale, noise=0.0, steps=STEPS):
    pos = pos0.copy()
    vel = vel0.copy()
    traj = [pos.copy()]
    for _ in range(steps):
        a = accelerations(pos, masses, G, dark_strength, dark_scale)
        vel += a * DT
        pos += vel * DT
        if noise > 0:
            pos += np.random.normal(0, noise, pos.shape) * DT
            vel += np.random.normal(0, noise/10, vel.shape) * DT
        traj.append(pos.copy())
    return np.array(traj)

def predict(pos0, vel0, masses, G, dark_strength, dark_scale, noise=0.0, steps=PREDICT_STEPS):
    pos = pos0.copy()
    vel = vel0.copy()
    pred = [pos.copy()]
    for _ in range(steps):
        a = accelerations(pos, masses, G, dark_strength, dark_scale)
        vel += a * DT
        pos += vel * DT
        if noise > 0:
            pos += np.random.normal(0, noise, pos.shape) * DT
            vel += np.random.normal(0, noise/10, vel.shape) * DT
        pred.append(pos.copy())
    return np.array(pred)

def mean_radius(traj):
    centers = np.mean(traj, axis=1, keepdims=True)
    radii = np.linalg.norm(traj - centers, axis=2)
    return np.mean(radii, axis=1)

# ========== ЗАПУСК ТРЁХ СЦЕНАРИЕВ ==========
print("Запуск симуляций...")
traj1 = evolve(pos0, vel0, masses, G, 0.0, DARK_SCALE, noise=0.0)
pred1 = predict(pos0, vel0, masses, G, 0.0, DARK_SCALE, noise=0.0)
mse1 = mean_squared_error(mean_radius(traj1[:PREDICT_STEPS+1]), mean_radius(pred1))

traj2 = evolve(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=0.0)
pred2 = predict(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=0.0)
mse2 = mean_squared_error(mean_radius(traj2[:PREDICT_STEPS+1]), mean_radius(pred2))

traj3 = evolve(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=NOISE_STRENGTH)
pred3 = predict(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=NOISE_STRENGTH)
mse3 = mean_squared_error(mean_radius(traj3[:PREDICT_STEPS+1]), mean_radius(pred3))

print(f"1. Только гравитация          → MSE = {mse1:.6f}")
print(f"2. Гравитация + тёмная энергия → MSE = {mse2:.6f}")
print(f"3. + случайный шум            → MSE = {mse3:.6f}")

# ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
rad1_real = mean_radius(traj1[:PREDICT_STEPS+1])
rad1_pred = mean_radius(pred1)
rad2_real = mean_radius(traj2[:PREDICT_STEPS+1])
rad2_pred = mean_radius(pred2)
rad3_real = mean_radius(traj3[:PREDICT_STEPS+1])
rad3_pred = mean_radius(pred3)
ax.plot(rad1_real, 'b-', label='Гравитация (реал)', lw=2)
ax.plot(rad1_pred, 'b--', label='Гравитация (предск)', alpha=0.7)
ax.plot(rad2_real, 'g-', label='+Тёмная энергия (реал)', lw=2)
ax.plot(rad2_pred, 'g--', label='+Тёмная энергия (предск)', alpha=0.7)
ax.plot(rad3_real, 'r-', label='+Шум (реал)', lw=2)
ax.plot(rad3_pred, 'r--', label='+Шум (предск)', alpha=0.7)
ax.set_xlabel('Шаг симуляции')
ax.set_ylabel('Средний радиус')
ax.set_title('Предсказуемость среднего радиуса')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True)

ax = axes[0,1]
ax.scatter(traj1[-1,:,0], traj1[-1,:,1], s=8, alpha=0.6, label='Гравитация')
ax.scatter(traj2[-1,:,0], traj2[-1,:,1], s=8, alpha=0.6, label='+Тёмная энергия')
ax.scatter(traj3[-1,:,0], traj3[-1,:,1], s=8, alpha=0.6, label='+Шум')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Финальное распределение частиц')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1,0]
labels = ['Гравитация', '+Тёмная\nэнергия', '+Шум']
mses = [mse1, mse2, mse3]
colors = ['blue', 'green', 'red']
ax.bar(labels, mses, color=colors)
ax.set_ylabel('Ошибка предсказания (MSE)')
ax.set_title('Сравнение предсказуемости')
ax.grid(True, axis='y')

ax = axes[1,1]
ax.plot(rad3_real, 'r-', label='Реальная эволюция', lw=2)
ax.plot(rad3_pred, 'b--', label='Предсказанная (из начала)', lw=2)
ax.set_xlabel('Шаг')
ax.set_ylabel('Средний радиус')
ax.set_title('Сценарий с шумом: реальный vs предсказанный')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=150)
plt.show()

print("\nРезультаты сохранены в 'simulation_results.png'")
"""
Симуляция Творения: предсказуемость эволюции Вселенной
Автор: Абдунодир (Ташкент)
Лицензия: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ========== ПАРАМЕТРЫ СИМУЛЯЦИИ ==========
NUM_PARTICLES = 120          # количество частиц
BOX_SIZE = 180               # начальный размер области
H0 = 0.02                    # постоянная Хаббла
G = 1.2                      # гравитационная постоянная
DARK_STRENGTH = 0.0008       # сила тёмной энергии
DARK_SCALE = 100.0           # масштаб тёмной энергии
NOISE_STRENGTH = 0.5         # сила случайных флуктуаций
DT = 0.05                    # шаг времени
STEPS = 400                  # всего шагов
PREDICT_STEPS = 60           # шагов для предсказания
np.random.seed(42)

# ========== СОЗДАНИЕ ЧАСТИЦ ==========
masses = np.random.uniform(0.5, 2.0, NUM_PARTICLES)
pos0 = np.random.uniform(-BOX_SIZE/2, BOX_SIZE/2, (NUM_PARTICLES, 2))
r0 = np.sqrt(pos0[:,0]**2 + pos0[:,1]**2)
vel0 = np.zeros_like(pos0)
for i in range(NUM_PARTICLES):
    if r0[i] > 0:
        vel0[i] = H0 * pos0[i]   # закон Хаббла
    vel0[i] += np.random.uniform(-0.1, 0.1, 2)  # небольшой шум

# ========== ФИЗИЧЕСКИЕ ФУНКЦИИ ==========
def accelerations(pos, masses, G, dark_strength, dark_scale):
    n = len(masses)
    a = np.zeros_like(pos)
    for i in range(n):
        for j in range(i+1, n):
            dr = pos[j] - pos[i]
            dist = np.linalg.norm(dr) + 0.5
            f_grav = G * masses[i] * masses[j] / dist**2
            f_dark = dark_strength * masses[i] * masses[j] * (dist / dark_scale)
            f_total = -f_grav + f_dark
            a[i] += f_total * dr / dist / masses[i]
            a[j] -= f_total * dr / dist / masses[j]
    return a

def evolve(pos0, vel0, masses, G, dark_strength, dark_scale, noise=0.0, steps=STEPS):
    pos = pos0.copy()
    vel = vel0.copy()
    traj = [pos.copy()]
    for _ in range(steps):
        a = accelerations(pos, masses, G, dark_strength, dark_scale)
        vel += a * DT
        pos += vel * DT
        if noise > 0:
            pos += np.random.normal(0, noise, pos.shape) * DT
            vel += np.random.normal(0, noise/10, vel.shape) * DT
        traj.append(pos.copy())
    return np.array(traj)

def predict(pos0, vel0, masses, G, dark_strength, dark_scale, noise=0.0, steps=PREDICT_STEPS):
    pos = pos0.copy()
    vel = vel0.copy()
    pred = [pos.copy()]
    for _ in range(steps):
        a = accelerations(pos, masses, G, dark_strength, dark_scale)
        vel += a * DT
        pos += vel * DT
        if noise > 0:
            pos += np.random.normal(0, noise, pos.shape) * DT
            vel += np.random.normal(0, noise/10, vel.shape) * DT
        pred.append(pos.copy())
    return np.array(pred)

def mean_radius(traj):
    centers = np.mean(traj, axis=1, keepdims=True)
    radii = np.linalg.norm(traj - centers, axis=2)
    return np.mean(radii, axis=1)

# ========== ЗАПУСК ТРЁХ СЦЕНАРИЕВ ==========
print("Запуск симуляций...")
traj1 = evolve(pos0, vel0, masses, G, 0.0, DARK_SCALE, noise=0.0)
pred1 = predict(pos0, vel0, masses, G, 0.0, DARK_SCALE, noise=0.0)
mse1 = mean_squared_error(mean_radius(traj1[:PREDICT_STEPS+1]), mean_radius(pred1))

traj2 = evolve(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=0.0)
pred2 = predict(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=0.0)
mse2 = mean_squared_error(mean_radius(traj2[:PREDICT_STEPS+1]), mean_radius(pred2))

traj3 = evolve(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=NOISE_STRENGTH)
pred3 = predict(pos0, vel0, masses, G, DARK_STRENGTH, DARK_SCALE, noise=NOISE_STRENGTH)
mse3 = mean_squared_error(mean_radius(traj3[:PREDICT_STEPS+1]), mean_radius(pred3))

print(f"1. Только гравитация          → MSE = {mse1:.6f}")
print(f"2. Гравитация + тёмная энергия → MSE = {mse2:.6f}")
print(f"3. + случайный шум            → MSE = {mse3:.6f}")

# ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
rad1_real = mean_radius(traj1[:PREDICT_STEPS+1])
rad1_pred = mean_radius(pred1)
rad2_real = mean_radius(traj2[:PREDICT_STEPS+1])
rad2_pred = mean_radius(pred2)
rad3_real = mean_radius(traj3[:PREDICT_STEPS+1])
rad3_pred = mean_radius(pred3)
ax.plot(rad1_real, 'b-', label='Гравитация (реал)', lw=2)
ax.plot(rad1_pred, 'b--', label='Гравитация (предск)', alpha=0.7)
ax.plot(rad2_real, 'g-', label='+Тёмная энергия (реал)', lw=2)
ax.plot(rad2_pred, 'g--', label='+Тёмная энергия (предск)', alpha=0.7)
ax.plot(rad3_real, 'r-', label='+Шум (реал)', lw=2)
ax.plot(rad3_pred, 'r--', label='+Шум (предск)', alpha=0.7)
ax.set_xlabel('Шаг симуляции')
ax.set_ylabel('Средний радиус')
ax.set_title('Предсказуемость среднего радиуса')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True)

ax = axes[0,1]
ax.scatter(traj1[-1,:,0], traj1[-1,:,1], s=8, alpha=0.6, label='Гравитация')
ax.scatter(traj2[-1,:,0], traj2[-1,:,1], s=8, alpha=0.6, label='+Тёмная энергия')
ax.scatter(traj3[-1,:,0], traj3[-1,:,1], s=8, alpha=0.6, label='+Шум')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Финальное распределение частиц')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1,0]
labels = ['Гравитация', '+Тёмная\nэнергия', '+Шум']
mses = [mse1, mse2, mse3]
colors = ['blue', 'green', 'red']
ax.bar(labels, mses, color=colors)
ax.set_ylabel('Ошибка предсказания (MSE)')
ax.set_title('Сравнение предсказуемости')
ax.grid(True, axis='y')

ax = axes[1,1]
ax.plot(rad3_real, 'r-', label='Реальная эволюция', lw=2)
ax.plot(rad3_pred, 'b--', label='Предсказанная (из начала)', lw=2)
ax.set_xlabel('Шаг')
ax.set_ylabel('Средний радиус')
ax.set_title('Сценарий с шумом: реальный vs предсказанный')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=150)
plt.show()

print("\nРезультаты сохранены в 'simulation_results.png'")
