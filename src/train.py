from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocess import build_preprocessor
from model import build_model
from utils import evaluate_model


def main():

    df = load_data("../boston_housing.csv")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.columns.tolist()
    categorical_features = []

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = build_model(preprocessor)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    evaluate_model(y_test, predictions)

if __name__ == "__main__":
    main()