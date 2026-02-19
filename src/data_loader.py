import pandas as pd

column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

def load_data(path):
    df = pd.read_csv(path, header=None, delimiter=r"\s+", names=column_names)
    return df

