import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import  mean_squared_error

class CaliforniaHousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None 
        self.y_pred = None

    def load_data(self):
        california = fetch_california_housing() 
        X, y = california.data, california.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return rmse

    def run(self):
        self.load_data()
        self.train_model()
        return self.evaluate_model()


class EnhancedCaliforniaHousePricePredictor(CaliforniaHousePricePredictor):
    def train_model_with_cross_validation(self):
        parameters = {'alpha': [0.1, 1, 10, 100]}
        ridge = Ridge()
        clf = GridSearchCV(ridge, parameters, cv=5)
        clf.fit(self.X_train, self.y_train)
        self.model = clf.best_estimator_
        print(f"Melhor parâmetro: {clf.best_params_}")
    
    def plot_feature_vs_price(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.X_train[:, 0], y=self.y_train)
        plt.xlabel('Media da Renda dos Moradores')
        plt.ylabel('Preço da Casa')
        plt.title('Preço da Casa vs Renda dos Moradores')
        plt.show()
    
    def plot_real_vs_predicted_prices(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        plt.xlabel('Preço Real')
        plt.ylabel('Preço Predito')
        plt.title('Preço Real vs Preço Predito')
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red')
        plt.show()        
    
    def run(self):
        self.load_data()
        self.train_model_with_cross_validation()
        rmse = self.evaluate_model()
        self.plot_feature_vs_price()
        self.plot_real_vs_predicted_prices()
        return rmse

    
if __name__ == "__main__":
    enhanced_predictor = EnhancedCaliforniaHousePricePredictor()
    rmse = enhanced_predictor.run()
    print(f"RSME: {rmse}")