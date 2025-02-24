import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv('cleaned_hh_ru_dataset.csv')
data_copy = data.copy()

# Группировка данных по профессиям и расчет средней ожидаемой зарплаты
avg_salary_by_profession = data.groupby('profession')['expected_salary'].mean().sort_values(ascending=False)
print(f'Группировка данных по профессиям и расчет средней ожидаемой зарплаты  {avg_salary_by_profession}')

# Визуализация
plt.figure(figsize=(12, 8))
avg_salary_by_profession.head(20).plot(kind='bar', color='skyblue')  # Топ-20 профессий
plt.title('Средняя ожидаемая зарплата по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()