# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib
from datetime import datetime


def load_data():

    df = pd.read_csv('housing.csv')

    
    df.dropna(inplace=True)

    return df


def train_polynomial_regression_model():
    df = load_data()


    X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
            'median_income']]
    y = df['median_house_value']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразование признаков в полиномиальные
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)


    model = LinearRegression()
    model.fit(X_train_poly, y_train)


    joblib.dump(model, 'polynomial_regression_model.pkl')
    joblib.dump(poly, 'polynomial_features_transformer.pkl')

    print("Модель полиномиальной регрессии успешно обучена и сохранена.")


if __name__ == '__main__':
    train_polynomial_regression_model()