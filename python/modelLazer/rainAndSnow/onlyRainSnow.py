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

# Параметры осадков
precipitation_type = 'rain'  # 'rain' или 'snow'
intensity_level = 'moderate'  # 'light', 'moderate', 'heavy'

# Фиксированные параметры среды
if precipitation_type == 'rain':
    if intensity_level == 'light':
        alpha = 0.001  # Коэффициент затухания [1/м]
        beta = omega / c * 1.0003  # Фазовая постоянная [рад/м]
    elif intensity_level == 'moderate':
        alpha = 0.005
        beta = omega / c * 1.0005
    else:
        alpha = 0.01
        beta = omega / c * 1.0008
else:  # snow
    if intensity_level == 'light':
        alpha = 0.0005
        beta = omega / c * 1.0002
    elif intensity_level == 'moderate':
        alpha = 0.002
        beta = omega / c * 1.0003
    else:
        alpha = 0.005
        beta = omega / c * 1.0005

# Расчет распространения поля
z = np.linspace(0, distance, num_points)
E_z = E0 * np.exp(-alpha * z) * np.exp(-1j * beta * z)

# Расчет интенсивности
intensity = 0.5 * c * epsilon_0 * np.abs(E_z)**2

# Временная зависимость интенсивности у лазера (z=0)
t = np.linspace(0, t_max, 100)
I0_t = 0.5 * c * epsilon_0 * E0**2 * np.ones_like(t)

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

# Создание фигуры с графиками
plt.figure(figsize=(12, 8))

# 1. Временная зависимость интенсивности у лазера
plt.subplot(2, 2, 1)
plt.plot(t*1e9, I0_t)
plt.title('Интенсивность у лазера')
plt.xlabel('Время (нс)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 2. Пространственное распределение интенсивности
plt.subplot(2, 2, 2)
plt.plot(z, intensity)
plt.title('Затухание интенсивности')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 3. Логарифмический масштаб для затухания
plt.subplot(2, 2, 3)
plt.semilogy(z, intensity)
plt.title('Затухание (лог. шкала)')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 4. Энергетический баланс
plt.subplot(2, 2, 4)
plt.text(0.1, 0.4,
        f"Энергетический баланс:\n"
        f"Тип осадков: {precipitation_type}\n"
        f"Интенсивность: {intensity_level}\n"
        f"Излучено: {E_initial/photon_energy_ev:.2e} эВ\n"
        f"Принято: {E_receiver/photon_energy_ev:.2e} эВ\n"
        f"Потери: {losses*100:.1f}%",
        bbox=dict(facecolor='white', alpha=0.8))
plt.axis('off')

plt.tight_layout(pad=2.0)

# Простое имя файла без данных массива
filename = f'laser_{precipitation_type}_{intensity_level}.png'
plt.savefig(filename, dpi=100, bbox_inches='tight')
plt.close()

# Вывод информации
print(f"Моделирование завершено в {datetime.now().strftime('%H:%M:%S')}")
print(f"Результаты сохранены в файл: {filename}")
print(f"Параметры среды: α = {alpha:.4e} 1/м, β = {beta:.4e} рад/м")
print(f"Начальная интенсивность: {I0:.4e} Вт/м²")
print(f"Интенсивность на приемнике: {I_receiver:.4e} Вт/м²")