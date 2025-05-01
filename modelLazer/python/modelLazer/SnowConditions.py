import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Параметры модели
c = 3e8  # скорость света (м/с)
L = 1000  # расстояние распространения (м)
Nz = 500  # количество точек по пространству
Nt = 1000  # количество временных шагов
dz = L / Nz
dt = dz / (2 * c)  # устойчивый шаг по времени

# Параметры сред для снега
mu0 = 4 * np.pi * 1e-7  # магнитная постоянная
epsilon_air = 1.0006  # диэлектрическая проницаемость воздуха
epsilon_snow = 1.5  # диэлектрическая проницаемость снега
sigma_light = 0.0001  # проводимость легкого снега (См/м)
sigma_heavy = 0.001  # проводимость сильного снега (См/м)

# Пространственная сетка
z = np.linspace(0, L, Nz)

# 1. Однородная среда (сильный снег постоянной интенсивности)
epsilon_homogeneous = np.ones(Nz) * epsilon_snow
sigma_homogeneous = np.ones(Nz) * sigma_heavy

# 2. Неоднородная среда (переменная интенсивность снега)
snow_intensity = 0.5 * (1 + np.sin(2 * np.pi * z / 150))  # периодическое изменение
epsilon_heterogeneous = epsilon_air + (epsilon_snow - epsilon_air) * snow_intensity
sigma_heterogeneous = sigma_light + (sigma_heavy - sigma_light) * snow_intensity


# Функция для моделирования
def simulate(epsilon_z, sigma_z, label):
    E = np.zeros(Nz)
    E_prev = np.zeros(Nz)

    # Источник (импульс 2.4 ГГц)
    freq = 2.4e9
    t = np.arange(Nt) * dt
    t0 = 1e-8
    sigma_t = 1e-9
    source = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2) * np.sin(2 * np.pi * freq * t)

    E_history = np.zeros((Nt, Nz))

    for k in range(Nt):
        E_next = np.zeros(Nz)
        for i in range(1, Nz - 1):
            d2E_dz2 = (E[i + 1] - 2 * E[i] + E[i - 1]) / dz ** 2
            dE_dt = (E[i] - E_prev[i]) / dt

            term1 = (epsilon_z[i] / c ** 2) * (E[i] - 2 * E_prev[i] + E_next[i]) / dt ** 2
            term2 = sigma_z[i] * mu0 * dE_dt
            E_next[i] = (d2E_dz2 - term1 - term2) * (c * dt) ** 2 / epsilon_z[i] + 2 * E[i] - E_prev[i]

        E_next[10] += source[k]
        E_next[0] = E[1]
        E_next[-1] = E[-2]

        E_prev, E = E, E_next
        E_history[k, :] = E

    # Анализ результатов
    observation_point = Nz // 2
    E_observed = E_history[:, observation_point]
    spectrum = np.abs(fft(E_observed))[:Nt // 2]
    freqs = fftfreq(Nt, dt)[:Nt // 2]

    amplitude = np.max(np.abs(E_history[:, 100:]), axis=0)
    distance = z[100:]

    return {
        'label': label,
        'z': z,
        'E': E,
        'E_observed': E_observed,
        't': t,
        'freqs': freqs,
        'spectrum': spectrum,
        'distance': distance,
        'amplitude': amplitude,
        'epsilon_z': epsilon_z,
        'sigma_z': sigma_z,
        'snow_intensity': snow_intensity if 'Неоднородный' in label else np.ones(Nz) * 0.7
    }


# Запуск моделирования
results = [
    simulate(epsilon_homogeneous, sigma_homogeneous, "Однородный сильный снег"),
    simulate(epsilon_heterogeneous, sigma_heterogeneous, "Неоднородный снегопад")
]

# Визуализация результатов
plt.figure(figsize=(16, 12))

# 1. Распределение поля и интенсивности снега
plt.subplot(2, 2, 1)
for res in results:
    plt.plot(res['z'], res['E'], label=f"Поле ({res['label']})")
    plt.plot(res['z'], res['snow_intensity'], '--', label=f"Интенсивность снега ({res['label']})")
plt.title('Распределение поля и интенсивности снега')
plt.xlabel('Расстояние (м)')
plt.ylabel('Нормированные величины')
plt.legend()
plt.grid(True)

# 2. Сигнал в точке наблюдения
plt.subplot(2, 2, 2)
for res in results:
    plt.plot(res['t'] * 1e9, res['E_observed'], label=res['label'])
plt.title('Сигнал в точке наблюдения (z={} м)'.format(L // 2))
plt.xlabel('Время (нс)')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

# 3. Спектры сигналов
plt.subplot(2, 2, 3)
for res in results:
    plt.plot(res['freqs'] / 1e9, 20 * np.log10(res['spectrum']), label=res['label'])
plt.title('Спектры сигналов')
plt.xlabel('Частота (ГГц)')
plt.ylabel('Амплитуда (дБ)')
plt.xlim(0, 5)
plt.legend()
plt.grid(True)

# 4. Параметры сред
plt.subplot(2, 2, 4)
plt.plot(results[0]['z'], results[0]['epsilon_z'], label='ε(z) - Однородный')
plt.plot(results[1]['z'], results[1]['epsilon_z'], label='ε(z) - Неоднородный')
plt.plot(results[0]['z'], results[0]['sigma_z'], label='σ(z) - Однородный')
plt.plot(results[1]['z'], results[1]['sigma_z'], label='σ(z) - Неоднородный')
plt.title('Параметры сред')
plt.xlabel('Расстояние (м)')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Дополнительные графики
plt.figure(figsize=(14, 5))

# 1. Затухание сигнала
plt.subplot(1, 2, 1)
for res in results:
    plt.plot(res['distance'], 20 * np.log10(res['amplitude']), label=res['label'])
plt.title('Затухание сигнала в снегу')
plt.xlabel('Расстояние (м)')
plt.ylabel('Амплитуда (дБ)')
plt.legend()
plt.grid(True)

# 2. Сравнение диэлектрических свойств
plt.subplot(1, 2, 2)
plt.plot(results[0]['z'], results[0]['epsilon_z'], label='Однородный снег (ε=1.5)')
plt.plot(results[1]['z'], results[1]['epsilon_z'], label='Неоднородный снег (ε=1.0-1.5)')
plt.title('Диэлектрическая проницаемость')
plt.xlabel('Расстояние (м)')
plt.ylabel('ε(z)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()