import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../hh_ru_dataset.csv')
data_copy = data.copy()

# Отчистка датасета от столбцов по которым не будет проводится анализ

data_copy = data_copy.drop(columns=['topic_id', 'resume_id','resume_skills_list', 'vacancy_id'])

# Топ-10 самых популярных профессий в резюме и вакансиях

# 1. Топ-10 самых популярных профессий в резюме
top_resume_professions = data_copy['profession'].value_counts().head(10)
print(f'Топ-10 самых популярных профессий в резюме: {top_resume_professions}')
print('_____________________________________________________________________')


# 2. Топ-10 самых популярных профессий в вакансиях
top_vacancy_professions = data_copy['vacancy_employment_type'].value_counts().head(10)
print(f'Топ-10 самых популярных профессий в вакансиях: {top_vacancy_professions}')
print('_____________________________________________________________________')

# 3. Соответствие профессий в резюме и вакансиях
profession_match = pd.crosstab(data_copy['profession'], data_copy['vacancy_employment_type'])
print(f'Соответствие профессий в резюме и вакансиях: {profession_match}')
print('_____________________________________________________________________')


# 4. Профессии с самыми высокими и самыми низкими зарплатами
# Средняя ожидаемая зарплата по профессиям в резюме
avg_salary_by_profession = data_copy.groupby('profession')['expected_salary'].mean().sort_values(ascending=False)
print(f'Средняя ожидаемая зарплата по профессиям в резюме: {avg_salary_by_profession}')
print('_____________________________________________________________________')

# Средняя предлагаемая зарплата по профессиям в вакансиях
avg_compensation_by_profession = data_copy.groupby('vacancy_employment_type')[['compensation_from', 'compensation_to']].mean()
avg_compensation_by_profession['avg_compensation'] = (avg_compensation_by_profession['compensation_from'] + avg_compensation_by_profession['compensation_to']) / 2
print(f'Средняя предлагаемая зарплата по профессиям в вакансиях: {avg_compensation_by_profession}')
print('_____________________________________________________________________')

# Визуализация
plt.figure(figsize=(15, 10))

# 1. Топ-10 самых популярных профессий в резюме
top_resume_professions = data_copy['profession'].value_counts().head(10)
print(f'Топ-10 самых популярных профессий в резюме: {top_resume_professions}')
print('_____________________________________________________________________')

# 2. Топ-10 самых популярных профессий в вакансиях
top_vacancy_professions = data_copy['vacancy_employment_type'].value_counts().head(10)
print(f'Топ-10 самых популярных профессий в вакансиях: {top_vacancy_professions}')
print('_____________________________________________________________________')

# 3. Профессии с самыми высокими зарплатами (резюме)
avg_salary_by_profession = data_copy.groupby('profession')['expected_salary'].mean().sort_values(ascending=False)
print(f'Профессии с самыми высокими зарплатами (резюме): {avg_salary_by_profession}')
print('_____________________________________________________________________')

# 4. Профессии с самыми высокими зарплатами (вакансии)
avg_compensation_by_profession = data_copy.groupby('vacancy_employment_type')[['compensation_from', 'compensation_to']].mean()
avg_compensation_by_profession['avg_compensation'] = (avg_compensation_by_profession['compensation_from'] + avg_compensation_by_profession['compensation_to']) / 2
print(f'Профессии с самыми высокими зарплатами (вакансии): {avg_compensation_by_profession}')
print('_____________________________________________________________________')

# График 1: Топ-10 профессий в резюме
plt.figure(figsize=(12, 6))  # Увеличиваем размер графика
top_resume_professions.plot(kind='bar', color='skyblue')
plt.title('Топ-10 профессий в резюме', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')  # Поворачиваем подписи на 45 градусов
plt.tight_layout()  # Автоматическая настройка layout
plt.show()

# График 2: Топ-10 профессий в вакансиях
plt.figure(figsize=(12, 6))  # Увеличиваем размер графика
top_vacancy_professions.plot(kind='bar', color='lightgreen')
plt.title('Топ-10 профессий в вакансиях', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')  # Поворачиваем подписи на 45 градусов
plt.tight_layout()  # Автоматическая настройка layout
plt.show()

# График 3: Профессии с самыми высокими зарплатами (резюме)
plt.figure(figsize=(12, 6))  # Увеличиваем размер графика
avg_salary_by_profession.head(10).plot(kind='bar', color='orange')
plt.title('Топ-10 профессий с самыми высокими зарплатами (резюме)', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')  # Поворачиваем подписи на 45 градусов
plt.tight_layout()  # Автоматическая настройка layout
plt.show()

# График 4: Профессии с самыми высокими зарплатами (вакансии)
plt.figure(figsize=(12, 6))  # Увеличиваем размер графика
avg_compensation_by_profession['avg_compensation'].sort_values(ascending=False).head(10).plot(kind='bar', color='purple')
plt.title('Топ-10 профессий с самыми высокими зарплатами (вакансии)', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя предлагаемая зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')  # Поворачиваем подписи на 45 градусов
plt.tight_layout()  # Автоматическая настройка layout
plt.show()