import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Анализ опыта работы: распределение по профессиям
experience_by_profession = data.groupby('profession')['work_experience_months'].mean().sort_values()

# Визуализация
plt.figure(figsize=(12, 8))
sns.barplot(x=experience_by_profession.index, y=experience_by_profession.values, palette='plasma')
plt.title('Средний опыт работы по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Опыт работы (месяцы)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Анализ опыта работы и ожидаемой зарплаты
plt.figure(figsize=(10, 6))
sns.scatterplot(x='work_experience_months', y='expected_salary', data=data, hue='profession', palette='viridis')
plt.title('Опыт работы vs Ожидаемая зарплата', fontsize=16)
plt.xlabel('Опыт работы (месяцы)', fontsize=14)
plt.ylabel('Ожидаемая зарплата', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()