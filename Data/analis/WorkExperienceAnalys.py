import pandas as pd
import matplotlib.pyplot as plt

# Настройка отображения чисел в Pandas (отключение научной нотации)
pd.set_option('display.float_format', '{:.2f}'.format)

# Загрузка данных
data = pd.read_csv('../hh_ru_dataset.csv')
data_copy = data.copy()

# Очистка датасета от столбцов, по которым не будет проводиться анализ
data_copy = data_copy.drop(columns=['topic_id', 'resume_id', 'resume_skills_list', 'vacancy_id'])

# 1. Распределение соискателей по опыту работы
# Преобразуем опыт работы из месяцев в категории
def categorize_experience(months):
    if months < 12:
        return 'Менее 1 года'
    elif 12 <= months < 36:
        return '1-3 года'
    elif 36 <= months < 60:
        return '3-5 лет'
    else:
        return 'Более 5 лет'

data_copy['experience_category'] = data_copy['work_experience_months'].apply(categorize_experience)

# Распределение соискателей по категориям опыта
experience_distribution = data_copy['experience_category'].value_counts(normalize=True) * 100
print("Распределение соискателей по опыту работы (%):")
print(experience_distribution.to_string())
print('_____________________________________________________________________')

# График: Распределение соискателей по опыту работы
plt.figure(figsize=(10, 6))
experience_distribution.plot(kind='bar', color='skyblue')
plt.title('Распределение соискателей по опыту работы', fontsize=16)
plt.xlabel('Опыт работы', fontsize=14)
plt.ylabel('Процент соискателей', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# 2. Влияние опыта работы на ожидаемые зарплаты
avg_salary_by_experience = data_copy.groupby('experience_category')['expected_salary'].mean().sort_values(ascending=False)
print("Средняя ожидаемая зарплата по опыту работы:")
print(avg_salary_by_experience.to_string())
print('_____________________________________________________________________')

# График: Влияние опыта работы на ожидаемые зарплаты
plt.figure(figsize=(10, 6))
avg_salary_by_experience.plot(kind='bar', color='orange')
plt.title('Влияние опыта работы на ожидаемые зарплаты', fontsize=16)
plt.xlabel('Опыт работы', fontsize=14)
plt.ylabel('Средняя ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# 3. Сравнение опыта работы в резюме и требований в вакансиях
# Опыт работы в резюме
resume_experience_distribution = data_copy['experience_category'].value_counts(normalize=True) * 100
print("Распределение опыта работы в резюме (%):")
print(resume_experience_distribution.to_string())
print('_____________________________________________________________________')

# Опыт работы в вакансиях (если есть столбец с требованиями)
if 'vacancy_experience' in data_copy.columns:
    vacancy_experience_distribution = data_copy['vacancy_experience'].value_counts(normalize=True) * 100
    print("Распределение требований к опыту работы в вакансиях (%):")
    print(vacancy_experience_distribution.to_string())
    print('_____________________________________________________________________')

    # График: Сравнение опыта работы в резюме и вакансиях
    plt.figure(figsize=(12, 6))
    resume_experience_distribution.plot(kind='bar', color='skyblue', label='Резюме')
    vacancy_experience_distribution.plot(kind='bar', color='lightgreen', label='Вакансии', alpha=0.7)
    plt.title('Сравнение опыта работы в резюме и вакансиях', fontsize=16)
    plt.xlabel('Опыт работы', fontsize=14)
    plt.ylabel('Процент', fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Столбец 'vacancy_experience' отсутствует в данных. Невозможно провести сравнение.")
    print('_____________________________________________________________________')