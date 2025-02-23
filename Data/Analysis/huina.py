import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных (предположим, что данные уже загружены в переменную data)
data = pd.read_csv('datas/cleaned_hh_ru_dataset.csv')

# 1. Средняя ожидаемая зарплата по профессиям
avg_salary_by_profession = data.groupby('profession')['expected_salary'].mean().sort_values(ascending=False)

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

# 2. Средняя зарплата по регионам
avg_salary_by_region = data.groupby('resume_region')['expected_salary'].mean().sort_values(ascending=False)

# Визуализация
plt.figure(figsize=(12, 8))
avg_salary_by_region.head(20).plot(kind='bar', color='lightgreen')  # Топ-20 регионов
plt.title('Средняя ожидаемая зарплата по регионам', fontsize=16)
plt.xlabel('Регион', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Средняя зарплата по уровню образования
avg_salary_by_education = data.groupby('education_level')['expected_salary'].mean().sort_values(ascending=False)

# Визуализация
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_salary_by_education.index, y=avg_salary_by_education.values, palette='viridis')
plt.title('Средняя ожидаемая зарплата по уровню образования', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Средняя зарплата по опыту работы
# Добавим столбец с опытом работы в годах
data['work_experience_years'] = data['work_experience_months'] / 12

# Группировка по опыту работы
avg_salary_by_experience = data.groupby('work_experience_years')['expected_salary'].mean()

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(avg_salary_by_experience.index, avg_salary_by_experience.values, marker='o', linestyle='-', color='orange')
plt.title('Средняя ожидаемая зарплата по опыту работы', fontsize=16)
plt.xlabel('Опыт работы (годы)', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Вилка зарплат в вакансиях (compensation_from и compensation_to)
# Средняя вилка зарплат по профессиям
data['average_compensation'] = (data['compensation_from'] + data['compensation_to']) / 2
avg_compensation_by_profession = data.groupby('profession')['average_compensation'].mean().sort_values(ascending=False)

# Визуализация
plt.figure(figsize=(12, 8))
avg_compensation_by_profession.head(20).plot(kind='bar', color='purple')  # Топ-20 профессий
plt.title('Средняя вилка зарплат по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя вилка зарплат', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()