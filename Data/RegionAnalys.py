import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('hh_ru_dataset.csv')
data_copy = data.copy()

# Отчистка датасета от столбцов по которым не будет проводится анализ

data_copy = data_copy.drop(columns=['topic_id', 'resume_id','resume_skills_list', 'vacancy_id'])

# Анализ регионов

# 1. Распределение резюме и вакансий по регионам
print('Распределение резюме и вакансий по регионам')
print('_____________________________________________________________________')
resume_by_region = data_copy['resume_region'].value_counts().head(10)  # Топ-10 регионов по резюме
print(f'Топ-10 регионов по резюме: {resume_by_region}')
print('_____________________________________________________________________')
vacancy_by_region = data_copy['vacancy_region'].value_counts().head(10)  # Топ-10 регионов по вакансиям
print(f'Топ-10 регионов по вакансиям: {vacancy_by_region}')
print('_____________________________________________________________________')

# 2. Сравнение ожидаемых и предлагаемых зарплат по регионам
# Средняя ожидаемая зарплата по регионам (из резюме)
avg_expected_salary_by_region = data_copy.groupby('resume_region')['expected_salary'].mean().sort_values(ascending=False)
print(f'Средняя ожидаемая зарплата по регионам (из резюме): {avg_expected_salary_by_region}')
print('_____________________________________________________________________')

# Средняя предлагаемая зарплата по регионам (из вакансий)
avg_compensation_by_region = data_copy.groupby('vacancy_region')[['compensation_from', 'compensation_to']].mean()
avg_compensation_by_region['avg_compensation'] = (avg_compensation_by_region['compensation_from'] + avg_compensation_by_region['compensation_to']) / 2
print(f'Средняя предлагаемая зарплата по регионам (из вакансий): {avg_compensation_by_region}')
print('_____________________________________________________________________')

# 3. Анализ готовности к переезду
relocation_stats = data_copy['relocation_status'].value_counts(normalize=True) * 100  # Процент готовых к переезду
top_relocation_regions = data_copy[data_copy['relocation_status'] == 'готов к переезду']['resume_region'].value_counts().head(10)  # Топ-10 регионов для переезда
print(f'Процент готовых к переезду: {relocation_stats}')
print('_____________________________________________________________________')
print(f'Топ-10 регионов для переезда: {relocation_stats}')
print('_____________________________________________________________________')


# Визуализация
plt.figure(figsize=(15, 18))

# График 1: Топ-10 регионов по количеству резюме
plt.figure(figsize=(10, 6))  # Создаем новое окно для графика
resume_by_region.plot(kind='bar', color='skyblue')
plt.title('Топ-10 регионов по количеству резюме', fontsize=14)
plt.xlabel('Регион', fontsize=12)
plt.ylabel('Количество резюме', fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.tight_layout()
plt.show()  # Отображаем график

# График 2: Топ-10 регионов по количеству вакансий
plt.figure(figsize=(10, 6))  # Создаем новое окно для графика
vacancy_by_region.plot(kind='bar', color='lightgreen')
plt.title('Топ-10 регионов по количеству вакансий', fontsize=14)
plt.xlabel('Регион', fontsize=12)
plt.ylabel('Количество вакансий', fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.tight_layout()
plt.show()  # Отображаем график

# График 3: Топ-10 регионов по ожидаемым зарплатам (резюме)
plt.figure(figsize=(10, 6))  # Создаем новое окно для графика
avg_expected_salary_by_region.head(10).plot(kind='bar', color='orange')
plt.title('Топ-10 регионов по ожидаемым зарплатам (резюме)', fontsize=14)
plt.xlabel('Регион', fontsize=12)
plt.ylabel('Средняя ожидаемая зарплата', fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.tight_layout()
plt.show()  # Отображаем график

# График 4: Топ-10 регионов по предлагаемым зарплатам (вакансии)
plt.figure(figsize=(10, 6))  # Создаем новое окно для графика
avg_compensation_by_region['avg_compensation'].sort_values(ascending=False).head(10).plot(kind='bar', color='purple')
plt.title('Топ-10 регионов по предлагаемым зарплатам (вакансии)', fontsize=14)
plt.xlabel('Регион', fontsize=12)
plt.ylabel('Средняя предлагаемая зарплата', fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.tight_layout()
plt.show()  # Отображаем график

# График 5: Готовность к переезду
plt.figure(figsize=(10, 6))  # Создаем новое окно для графика
relocation_stats.plot(kind='bar', color='teal')
plt.title('Готовность к переезду', fontsize=14)
plt.xlabel('Готовность к переезду', fontsize=12)
plt.ylabel('Процент соискателей', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()  # Отображаем график

# График 6: Топ-10 регионов для переезда
if not top_relocation_regions.empty:
    plt.figure(figsize=(10, 6))  # Создаем новое окно для графика
    top_relocation_regions.plot(kind='bar', color='brown')
    plt.title('Топ-10 регионов для переезда', fontsize=14)
    plt.xlabel('Регион', fontsize=12)
    plt.ylabel('Количество соискателей', fontsize=12)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.tight_layout()
    plt.show()  # Отображаем график
else:
    print("Нет данных для построения графика 'Топ-10 регионов для переезда'.")