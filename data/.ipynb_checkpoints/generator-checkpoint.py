import pandas as pd


data = {
    'height': [95, 102, 88, 110, 97, 105, 92, 100, 99, 108],
    'weight': [180, 210, 160, 230, 175, 220, 170, 200, 185, 225],
    'age': [5, 7, 4, 8, 6, 7, 5, 6, 5, 8],
    'tail_length': [90, 95, 85, 100, 92, 98, 88, 96, 91, 99],
    'sex': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)
df.to_csv('tiger.csv', index=False)