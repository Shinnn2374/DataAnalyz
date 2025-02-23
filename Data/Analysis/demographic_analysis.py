import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Гендерный анализ: распределение профессий по полу
gender_profession = data.groupby(['profession', 'gender']).size().unstack()

# Визуализация
plt.figure(figsize=(12, 8))
gender_profession.plot(kind='bar', stacked=True)
plt.title('Распределение профессий по полу', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Пол')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Возрастной анализ: возраст в зависимости от профессии
age_by_profession = data.groupby('profession')['age'].mean().sort_values()

plt.figure(figsize=(12, 8))
sns.boxplot(x='profession', y='age', data=data, palette='viridis')
plt.title('Распределение возраста по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Возраст', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Анализ уровня образования: влияние на ожидаемую зарплату
salary_by_education = data.groupby('education_level')['expected_salary'].mean().sort_values()

plt.figure(figsize=(12, 8))
sns.barplot(x=salary_by_education.index, y=salary_by_education.values, palette='coolwarm')
plt.title('Ожидаемая зарплата по уровню образования', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()