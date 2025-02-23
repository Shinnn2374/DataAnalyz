import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'datas/cleaned_hh_ru_dataset.csv'
data = pd.read_csv(file_path)

# Прогнозирование ожидаемой зарплаты
X = data[['age', 'work_experience_months', 'education_level']]
X = pd.get_dummies(X, columns=['education_level'], drop_first=True)
y = data['expected_salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(y_test, model.predict(X_test), alpha=0.5)
plt.title('Прогноз ожидаемой зарплаты', fontsize=16)
plt.xlabel('Фактическая зарплата', fontsize=14)
plt.ylabel('Прогнозируемая зарплата', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()