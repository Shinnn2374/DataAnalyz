import pandas as pd
import matplotlib.pyplot as plt

# Настройка отображения чисел в Pandas (отключение научной нотации)
pd.set_option('display.float_format', '{:.2f}'.format)

# Загрузка данных
data = pd.read_csv('hh_ru_dataset.csv')
data_copy = data.copy()

# Очистка датасета от столбцов, по которым не будет проводиться анализ
data_copy = data_copy.drop(columns=['topic_id', 'resume_id', 'resume_skills_list', 'vacancy_id'])

# 1. Распределение соискателей по уровню образования
education_distribution = data_copy['education_level'].value_counts(normalize=True) * 100
print("Распределение соискателей по уровню образования (%):")
print(education_distribution.to_string())
print('_____________________________________________________________________')

# График: Распределение соискателей по уровню образования
plt.figure(figsize=(10, 6))
education_distribution.plot(kind='bar', color='skyblue')
plt.title('Распределение соискателей по уровню образования', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Процент соискателей', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# 2. Влияние уровня образования на ожидаемые зарплаты
avg_salary_by_education = data_copy.groupby('education_level')['expected_salary'].mean().sort_values(ascending=False)
print("Средняя ожидаемая зарплата по уровню образования:")
print(avg_salary_by_education.to_string())
print('_____________________________________________________________________')

# График: Влияние уровня образования на ожидаемые зарплаты
plt.figure(figsize=(10, 6))
avg_salary_by_education.plot(kind='bar', color='orange')
plt.title('Влияние уровня образования на ожидаемые зарплаты', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Средняя ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# 3. Сравнение уровня образования в резюме и требований в вакансиях
# Уровень образования в резюме
resume_education_distribution = data_copy['education_level'].value_counts(normalize=True) * 100
print("Распределение уровня образования в резюме (%):")
print(resume_education_distribution.to_string())
print('_____________________________________________________________________')

# Уровень образования в вакансиях (если есть столбец с требованиями)
if 'vacancy_education_level' in data_copy.columns:
    vacancy_education_distribution = data_copy['vacancy_education_level'].value_counts(normalize=True) * 100
    print("Распределение требований к образованию в вакансиях (%):")
    print(vacancy_education_distribution.to_string())
    print('_____________________________________________________________________')

    # График: Сравнение уровня образования в резюме и вакансиях
    plt.figure(figsize=(12, 6))
    resume_education_distribution.plot(kind='bar', color='skyblue', label='Резюме')
    vacancy_education_distribution.plot(kind='bar', color='lightgreen', label='Вакансии', alpha=0.7)
    plt.title('Сравнение уровня образования в резюме и вакансиях', fontsize=16)
    plt.xlabel('Уровень образования', fontsize=14)
    plt.ylabel('Процент', fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Столбец 'vacancy_education_level' отсутствует в данных. Невозможно провести сравнение.")
    print('_____________________________________________________________________')