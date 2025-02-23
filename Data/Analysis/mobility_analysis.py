import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Анализ готовности к переезду
relocation_status = data['relocation_status'].value_counts()

# Визуализация
plt.figure(figsize=(8, 6))
sns.barplot(x=relocation_status.index, y=relocation_status.values, palette='coolwarm')
plt.title('Готовность к переезду', fontsize=16)
plt.xlabel('Статус переезда', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Анализ готовности к командировкам
business_trip_readiness = data['business_trip_readiness'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=business_trip_readiness.index, y=business_trip_readiness.values, palette='magma')
plt.title('Готовность к командировкам', fontsize=16)
plt.xlabel('Готовность', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()