import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Параметры модели
c = 3e8  # скорость света (м/с)
L = 1e-6  # длина резонатора (м)
Nz = 500  # число точек по пространству
Nt = 1000  # число временных шагов
dz = L / Nz  # пространственный шаг
dt = dz / (2 * c)  # временной шаг (устойчивость)

# Определение сред
z = np.linspace(0, L, Nz)
n_vacuum = np.ones(Nz)

# Выбор среды для моделирования (изменить при необходимости)
n = n_vacuum

# Инициализация поля
E = np.zeros(Nz)
E_prev = np.zeros(Nz)
E_next = np.zeros(Nz)

# Параметры источника
source_pos = Nz // 2
t0 = 50 * dt
sigma = 10 * dt

# Коэффициент усиления (для моделирования лазера)
g = 100  # 1/м
use_gain = False  # Флаг для включения/выключения усиления

# Моделирование
for k in range(Nt):
    t = k * dt

    # Источник (гауссов импульс)
    pulse = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    E[source_pos] += pulse

    # Обновление поля
    for i in range(1, Nz - 1):
        E_next[i] = 2 * E[i] - E_prev[i] + (c * dt / (n[i] * dz)) ** 2 * (E[i + 1] - 2 * E[i] + E[i - 1])

        # Добавление усиления (если включено)
        if use_gain:
            E_next[i] += g * E[i] * dt

    # Граничные условия (поглощающие)
    E_next[0] = E[1]
    E_next[-1] = E[-2]

    E_prev, E = E, E_next

# Спектральный анализ
point_pos = Nz // 4  # Точка наблюдения
E_point = np.array([E[point_pos] for _ in range(Nt)])  # Для простоты используем последнее состояние
freq = fftfreq(Nt, dt)[:Nt // 2]
spectrum = np.abs(fft(E_point)[:Nt // 2])

# Создание итогового графика
plt.figure(figsize=(12, 8))

# 1. Распространение волны в последний момент времени
plt.subplot(2, 2, 1)
plt.plot(z, E)
plt.xlabel('Положение, z (м)')
plt.ylabel('Электрическое поле, E')
plt.title('Распределение поля в последний момент')
plt.grid(True)

# 2. Спектральный анализ
plt.subplot(2, 2, 2)
plt.plot(freq, spectrum)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда спектра')
plt.title('Спектр в точке наблюдения')
plt.grid(True)

# 3. Распределение показателя преломления
plt.subplot(2, 2, 3)
plt.plot(z, n)
plt.xlabel('z (м)')
plt.ylabel('n(z)')
plt.title('Распределение показателя преломления')
plt.grid(True)

# 4. Информация о моделировании
plt.subplot(2, 2, 4)
plt.axis('off')
plt.text(0.1, 0.8, f"Длина резонатора: {L:.1e} м", fontsize=10)
plt.text(0.1, 0.6, f"Шаг по пространству: {dz:.1e} м", fontsize=10)
plt.text(0.1, 0.4, f"Шаг по времени: {dt:.1e} с", fontsize=10)
plt.text(0.1, 0.2, f"Число временных шагов: {Nt}", fontsize=10)
plt.text(0.1, 0.0, f"Тип среды: {'разнородная' if np.any(n != n[0]) else 'однородная'}", fontsize=10)

plt.tight_layout()
plt.show()