import pandas as pd
import matplotlib.pyplot as plt

# Настройка отображения чисел в Pandas (отключение научной нотации)
pd.set_option('display.float_format', '{:.2f}'.format)

# Загрузка данных
data = pd.read_csv('../hh_ru_dataset.csv')
data_copy = data.copy()

# Очистка датасета от столбцов, по которым не будет проводиться анализ
data_copy = data_copy.drop(columns=['topic_id', 'resume_id', 'resume_skills_list', 'vacancy_id'])

# 1. Распределение соискателей по возрасту
# Преобразуем год рождения в возраст
from datetime import datetime
current_year = datetime.now().year
data_copy['age'] = current_year - data_copy['year_of_birth']

# Категоризация возраста
def categorize_age(age):
    if age < 25:
        return '18-25'
    elif 25 <= age < 35:
        return '25-35'
    elif 35 <= age < 45:
        return '35-45'
    else:
        return '45+'

data_copy['age_category'] = data_copy['age'].apply(categorize_age)

# Распределение соискателей по возрастным категориям
age_distribution = data_copy['age_category'].value_counts(normalize=True) * 100
print("Распределение соискателей по возрасту (%):")
print(age_distribution.to_string())
print('_____________________________________________________________________')

# График: Распределение соискателей по возрасту
plt.figure(figsize=(10, 6))
age_distribution.plot(kind='bar', color='skyblue')
plt.title('Распределение соискателей по возрасту', fontsize=16)
plt.xlabel('Возрастная категория', fontsize=14)
plt.ylabel('Процент соискателей', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# 2. Влияние возраста на ожидаемые зарплаты
avg_salary_by_age = data_copy.groupby('age_category')['expected_salary'].mean().sort_values(ascending=False)
print("Средняя ожидаемая зарплата по возрасту:")
print(avg_salary_by_age.to_string())
print('_____________________________________________________________________')

# График: Влияние возраста на ожидаемые зарплаты
plt.figure(figsize=(10, 6))
avg_salary_by_age.plot(kind='bar', color='orange')
plt.title('Влияние возраста на ожидаемые зарплаты', fontsize=16)
plt.xlabel('Возрастная категория', fontsize=14)
plt.ylabel('Средняя ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# 3. Распределение соискателей по полу
gender_distribution = data_copy['gender'].value_counts(normalize=True) * 100
print("Распределение соискателей по полу (%):")
print(gender_distribution.to_string())
print('_____________________________________________________________________')

# График: Распределение соискателей по полу
plt.figure(figsize=(8, 6))
gender_distribution.plot(kind='bar', color=['skyblue', 'lightpink'])
plt.title('Распределение соискателей по полу', fontsize=16)
plt.xlabel('Пол', fontsize=14)
plt.ylabel('Процент соискателей', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()

# 4. Сравнение ожидаемых зарплат по полу
avg_salary_by_gender = data_copy.groupby('gender')['expected_salary'].mean()
print("Средняя ожидаемая зарплата по полу:")
print(avg_salary_by_gender.to_string())
print('_____________________________________________________________________')

# График: Сравнение ожидаемых зарплат по полу
plt.figure(figsize=(8, 6))
avg_salary_by_gender.plot(kind='bar', color=['skyblue', 'lightpink'])
plt.title('Сравнение ожидаемых зарплат по полу', fontsize=16)
plt.xlabel('Пол', fontsize=14)
plt.ylabel('Средняя ожидаемая зарплата', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()