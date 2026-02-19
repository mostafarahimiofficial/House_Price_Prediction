from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def build_model(preprocessor):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    return pipeline