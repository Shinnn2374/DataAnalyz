import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Анализ времени от создания резюме до изменения состояния
data['resume_creation_date'] = pd.to_datetime(data['resume_creation_date'])
data['topic_creation_date'] = pd.to_datetime(data['topic_creation_date'])
data['time_to_change'] = (data['topic_creation_date'] - data['resume_creation_date']).dt.days

# Визуализация
plt.figure(figsize=(10, 6))
sns.histplot(data['time_to_change'], bins=30, kde=True, color='blue')
plt.title('Время от создания резюме до изменения состояния', fontsize=16)
plt.xlabel('Дни', fontsize=14)
plt.ylabel('Частота', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()