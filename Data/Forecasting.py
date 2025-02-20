import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка отображения чисел в Pandas (отключение научной нотации)
pd.set_option('display.float_format', '{:.2f}'.format)

# Загрузка данных
data = pd.read_csv('hh_ru_dataset.csv')
data_copy = data.copy()

# Очистка датасета от столбцов, по которым не будет проводиться анализ
data_copy = data_copy.drop(columns=['topic_id', 'resume_id', 'resume_skills_list', 'vacancy_id'])

# 1. Прогноз спроса на профессии
# Текущий спрос на профессии (топ-10 самых востребованных профессий)
top_professions = data_copy['profession'].value_counts().head(10)
print("Текущий спрос на профессии (топ-10):")
print(top_professions.to_string())
print('_____________________________________________________________________')

# График: Текущий спрос на профессии
plt.figure(figsize=(12, 6))
top_professions.plot(kind='bar', color='skyblue')
plt.title('Текущий спрос на профессии (топ-10)', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Количество вакансий', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# Прогноз: Предположим, что текущие тренды сохранятся
print("Прогноз: Наиболее востребованные профессии в ближайшие годы будут теми же, что и сейчас.")
print('_____________________________________________________________________')

# 2. Прогноз уровня зарплат
# Текущие средние зарплаты по профессиям
avg_salary_by_profession = data_copy.groupby('profession')['expected_salary'].mean().sort_values(ascending=False).head(10)
print("Текущие средние зарплаты по профессиям (топ-10):")
print(avg_salary_by_profession.to_string())
print('_____________________________________________________________________')

# График: Текущие средние зарплаты по профессиям
plt.figure(figsize=(12, 6))
avg_salary_by_profession.plot(kind='bar', color='orange')
plt.title('Текущие средние зарплаты по профессиям (топ-10)', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()

# Прогноз: Предположим, что зарплаты будут расти на 5% ежегодно
growth_rate = 0.05  # 5% рост
avg_salary_by_profession_future = avg_salary_by_profession * (1 + growth_rate) ** 3  # Прогноз на 3 года
print("Прогноз средних зарплат по профессиям через 3 года (рост на 5% ежегодно):")
print(avg_salary_by_profession_future.to_string())
print('_____________________________________________________________________')

# График: Прогноз средних зарплат по профессиям через 3 года
plt.figure(figsize=(12, 6))
avg_salary_by_profession_future.plot(kind='bar', color='lightgreen')
plt.title('Прогноз средних зарплат по профессиям через 3 года', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.tight_layout()
plt.show()