import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Кластеризация по возрасту и ожидаемой зарплате
X = data[['age', 'expected_salary']]
kmeans = KMeans(n_clusters=3)
data['cluster'] = kmeans.fit_predict(X)

# Визуализация
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='expected_salary', hue='cluster', data=data, palette='viridis')
plt.title('Кластеризация по возрасту и ожидаемой зарплате', fontsize=16)
plt.xlabel('Возраст', fontsize=14)
plt.ylabel('Ожидаемая зарплата', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()