import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, hbar, epsilon_0, mu_0
from datetime import datetime

# Параметры модели
wavelength = 850e-9  # Длина волны [м]
frequency = c / wavelength
omega = 2 * np.pi * frequency
E0 = 10.0  # Амплитуда поля [В/м]
distance = 1000.0  # Расстояние [м]
beam_diameter = 0.002  # Диаметр пучка [м]
beam_area = np.pi * (beam_diameter/2)**2  # Площадь пучка [м²]
t_max = 10e-9  # Время моделирования [с]
num_points = 500  # Число точек расчета

# Функции для неоднородной среды
def n_air(z):
    return 1.000293 + 2e-6 * np.sin(0.002 * z)

def sigma_air(z):
    return 5e-15 * (1 + 0.1 * np.cos(0.001 * z))

# Расчет параметров распространения
z = np.linspace(0, distance, num_points)
n = n_air(z)
sigma = sigma_air(z)
k = (omega * n) / c
alpha = (mu_0 * sigma * c**2) / (2 * n)

# Моделирование поля
E_z = E0 * np.exp(-alpha * z) * np.exp(-1j * k * z)

# Расчет интенсивности
intensity = 0.5 * c * epsilon_0 * n * np.abs(E_z)**2

# Временная зависимость интенсивности у лазера (z=0)
t = np.linspace(0, t_max, 100)
I0_t = 0.5 * c * epsilon_0 * n[0] * np.abs(E0 * np.exp(1j * omega * t))**2

# Энергетические расчеты
photon_energy_ev = (hbar * omega) / e  # Энергия фотона [эВ]
I0 = intensity[0]  # Начальная интенсивность
I_receiver = intensity[-1]  # Интенсивность на приемнике

# Мощности [Вт]
P_initial = I0 * beam_area
P_receiver = I_receiver * beam_area

# Энергии [эВ]
E_initial = (P_initial * t_max) / e
E_receiver = (P_receiver * t_max) / e

# Коэффициенты
transmission = P_receiver / P_initial
losses = 1 - transmission

# Создание фигуры с 4 графиками
plt.figure(figsize=(18, 12))

# 1. Временная зависимость интенсивности у лазера
plt.subplot(2, 2, 1)
plt.plot(t*1e9, I0_t)
plt.title('Временная зависимость интенсивности у лазера (z=0)')
plt.xlabel('Время (нс)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 2. Пространственное распределение интенсивности
plt.subplot(2, 2, 2)
plt.plot(z, intensity)
plt.title('Затухание интенсивности вдоль трассы')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 3. Логарифмический масштаб для затухания
plt.subplot(2, 2, 3)
plt.semilogy(z, intensity)
plt.title('Затухание интенсивности (логарифмическая шкала)')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 4. Энергетический баланс
plt.subplot(2, 2, 4)
plt.text(0.1, 0.5,
        f"Энергетический баланс:\n\n"
        f"Излученная энергия: {E_initial/photon_energy_ev:.2e} эВ\n"
        f"Энергия на приемнике: {E_receiver/photon_energy_ev:.2e} эВ\n"
        f"Потери энергии: {(E_initial-E_receiver)/photon_energy_ev:.2e} эВ\n"
        f"Коэффициент пропускания: {transmission*100:.2f}%\n"
        f"Потери: {losses*100:.2f}%",
        bbox=dict(facecolor='white', alpha=0.8))
plt.axis('off')

plt.tight_layout()
plt.show()

# Вывод дополнительной информации
print(f"Моделирование завершено в {datetime.now().strftime('%H:%M:%S')}")
print(f"Энергия фотона: {photon_energy_ev:.4f} эВ")
print(f"Начальная интенсивность: {I0:.4e} Вт/м²")
print(f"Интенсивность на приемнике: {I_receiver:.4e} Вт/м²")
print(f"Коэффициент затухания: {np.mean(alpha):.4e} 1/м")