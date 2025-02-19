import pandas as pd
from pandas import pivot

data = pd.read_csv('hh_ru_dataset.csv')
data_copy = data.copy()

# Отчистка датасета от столбцов по которым не будет проводится анализ

data_copy = data_copy.drop(columns=['topic_id', 'resume_id','resume_skills_list', 'vacancy_id'])
print(data_copy.columns)
