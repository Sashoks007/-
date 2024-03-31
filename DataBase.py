import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def load_data():

    df = pd.read_csv('housing.csv')


    df.dropna(inplace=True)

    return df

def train_and_save_model():
    df = load_data()

    # Выборка признаков и целевой переменной
    X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
    y = df['median_house_value']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

    # Сохранение модели
    joblib.dump(model, 'linear_regression_model.pkl')
    print("Модель успешно сохранена.")

if __name__ == '__main__':
    train_and_save_model()