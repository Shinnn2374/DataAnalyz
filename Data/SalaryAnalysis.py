import pandas as pd
import matplotlib.pyplot as plt

# Настройка отображения чисел в Pandas (отключение научной нотации)
pd.set_option('display.float_format', '{:.2f}'.format)

# Загрузка данных
data = pd.read_csv('hh_ru_dataset.csv')
data_copy = data.copy()

# Очистка датасета от столбцов, по которым не будет проводиться анализ
data_copy = data_copy.drop(columns=['topic_id', 'resume_id', 'resume_skills_list', 'vacancy_id'])

# Анализ зарплат
# 1. Сравнение ожидаемых и предлагаемых зарплат
# Средняя ожидаемая зарплата (из резюме)
avg_expected_salary = data_copy['expected_salary'].mean()
print(f'Средняя ожидаемая зарплата (из резюме): {avg_expected_salary:.2f}')
print('_____________________________________________________________________')


# Средняя предлагаемая зарплата (из вакансий)
avg_compensation = (data_copy['compensation_from'].mean() + data_copy['compensation_to'].mean()) / 2
print(f'Средняя предлагаемая зарплата (из вакансий): {avg_compensation:.2f}')
print('_____________________________________________________________________')


# Сравнение в виде графика
plt.figure(figsize=(8, 6))
plt.bar(['Ожидаемая зарплата', 'Предлагаемая зарплата'], [avg_expected_salary, avg_compensation], color=['skyblue', 'lightgreen'])
plt.title('Сравнение ожидаемых и предлагаемых зарплат', fontsize=16)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.show()

# 2. Средние зарплаты по профессиям и регионам
# Средняя ожидаемая зарплата по профессиям
avg_salary_by_profession = data_copy.groupby('profession')['expected_salary'].mean().sort_values(ascending=False).head(10)
print('Средняя ожидаемая зарплата по профессиям:')
print(avg_salary_by_profession.to_string())
print('_____________________________________________________________________')


# Средняя предлагаемая зарплата по профессиям
avg_compensation_by_profession = data_copy.groupby('vacancy_employment_type')[['compensation_from', 'compensation_to']].mean()
avg_compensation_by_profession['avg_compensation'] = (avg_compensation_by_profession['compensation_from'] + avg_compensation_by_profession['compensation_to']) / 2
avg_compensation_by_profession = avg_compensation_by_profession['avg_compensation'].sort_values(ascending=False).head(10)
print('Средняя предлагаемая зарплата по профессиям:')
print(avg_compensation_by_profession.to_string())
print('_____________________________________________________________________')


# Средняя ожидаемая зарплата по регионам
avg_salary_by_region = data_copy.groupby('resume_region')['expected_salary'].mean().sort_values(ascending=False).head(10)
print('Средняя ожидаемая зарплата по регионам:')
print(avg_salary_by_region.to_string())
print('_____________________________________________________________________')


# Средняя предлагаемая зарплата по регионам
avg_compensation_by_region = data_copy.groupby('vacancy_region')[['compensation_from', 'compensation_to']].mean()
avg_compensation_by_region['avg_compensation'] = (avg_compensation_by_region['compensation_from'] + avg_compensation_by_region['compensation_to']) / 2
avg_compensation_by_region = avg_compensation_by_region['avg_compensation'].sort_values(ascending=False).head(10)
print('Средняя предлагаемая зарплата по регионам:')
print(avg_compensation_by_region.to_string())
print('_____________________________________________________________________')


# График: Средние зарплаты по профессиям
plt.figure(figsize=(12, 6))
avg_salary_by_profession.plot(kind='bar', color='skyblue', label='Ожидаемая зарплата')
avg_compensation_by_profession.plot(kind='bar', color='lightgreen', label='Предлагаемая зарплата')
plt.title('Средние зарплаты по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# График: Средние зарплаты по регионам
plt.figure(figsize=(12, 6))
avg_salary_by_region.plot(kind='bar', color='skyblue', label='Ожидаемая зарплата')
avg_compensation_by_region.plot(kind='bar', color='lightgreen', label='Предлагаемая зарплата')
plt.title('Средние зарплаты по регионам', fontsize=16)
plt.xlabel('Регион', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Зарплаты в зависимости от уровня образования
# Средняя ожидаемая зарплата по уровню образования
avg_salary_by_education = data_copy.groupby('education_level')['expected_salary'].mean().sort_values(ascending=False)
print('Средняя ожидаемая зарплата по уровню образования:')
print(avg_salary_by_education.to_string())
print('_____________________________________________________________________')


# Средняя предлагаемая зарплата по уровню образования
avg_compensation_by_education = data_copy.groupby('education_level')[['compensation_from', 'compensation_to']].mean()
avg_compensation_by_education['avg_compensation'] = (avg_compensation_by_education['compensation_from'] + avg_compensation_by_education['compensation_to']) / 2
avg_compensation_by_education = avg_compensation_by_education['avg_compensation'].sort_values(ascending=False)
print('Средняя предлагаемая зарплата по уровню образования:')
print(avg_compensation_by_education.to_string())
print('_____________________________________________________________________')


# График: Зарплаты в зависимости от уровня образования
plt.figure(figsize=(12, 6))
avg_salary_by_education.plot(kind='bar', color='skyblue', label='Ожидаемая зарплата')
avg_compensation_by_education.plot(kind='bar', color='lightgreen', label='Предлагаемая зарплата')
plt.title('Зарплаты в зависимости от уровня образования', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.legend()
plt.tight_layout()
plt.show()