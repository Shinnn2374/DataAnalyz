import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = ('datas/cleaned_hh_ru_dataset.csv')
data = pd.read_csv(file_path)

# Просмотр первых строк данных для проверки
print("Первые строки данных:")
print(data.head())
print('__________________________________________________')

# Гендерный анализ: распределение профессий по полу
print("Гендерный анализ: распределение профессий по полу")
gender_profession = data.groupby(['profession', 'gender']).size().unstack()
print(gender_profession)
print('__________________________________________________')

# Возрастной анализ: возраст в зависимости от профессии
print("Возрастной анализ: возраст в зависимости от профессии")
age_by_profession = data.groupby('profession')['age'].mean().sort_values()
print(age_by_profession)
print('__________________________________________________')

# Визуализация возраста по профессиям
plt.figure(figsize=(40, 8))
sns.boxplot(x='profession', y='age', data=data, palette='viridis')
plt.title('Распределение возраста по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Возраст', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Поворот подписей для удобства
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Возрастной анализ: возраст в зависимости от региона
print("Возрастной анализ: возраст в зависимости от региона")
age_by_region = data.groupby('resume_region')['age'].mean().sort_values()
print(age_by_region)
print('__________________________________________________')

# Визуализация возраста по регионам (топ-10 регионов для читаемости)
top_regions = age_by_region.nlargest(10).index
data_top_regions = data[data['resume_region'].isin(top_regions)]

plt.figure(figsize=(12, 8))
sns.boxplot(x='resume_region', y='age', data=data_top_regions, palette='magma')
plt.title('Распределение возраста по регионам (топ-10)', fontsize=16)
plt.xlabel('Регион', fontsize=14)
plt.ylabel('Возраст', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Поворот подписей для удобства
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Анализ уровня образования: влияние на ожидаемую зарплату
print("Анализ уровня образования: влияние на ожидаемую зарплату")
salary_by_education = data.groupby('education_level')['expected_salary'].mean().sort_values()
print(salary_by_education)
print('__________________________________________________')

# Визуализация ожидаемой зарплаты по уровню образования
plt.figure(figsize=(12, 8))
sns.barplot(x=salary_by_education.index, y=salary_by_education.values, palette='coolwarm')
plt.title('Ожидаемая зарплата по уровню образования', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Поворот подписей для удобства
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()