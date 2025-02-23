import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Анализ предпочтений по графику работы
work_schedule_distribution = data['work_schedule'].value_counts()

# Визуализация
plt.figure(figsize=(10, 6))
sns.barplot(x=work_schedule_distribution.index, y=work_schedule_distribution.values, palette='coolwarm')
plt.title('Распределение предпочтений по графику работы', fontsize=16)
plt.xlabel('График работы', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()