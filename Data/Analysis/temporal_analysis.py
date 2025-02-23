import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Преобразование дат
data['resume_creation_date'] = pd.to_datetime(data['resume_creation_date'])
data['vacancy_creation_date'] = pd.to_datetime(data['vacancy_creation_date'])

# Анализ динамики создания резюме
resume_creation_trend = data.resample('M', on='resume_creation_date').size()

# Визуализация
plt.figure(figsize=(12, 6))
resume_creation_trend.plot()
plt.title('Динамика создания резюме', fontsize=16)
plt.xlabel('Дата', fontsize=14)
plt.ylabel('Количество резюме', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()