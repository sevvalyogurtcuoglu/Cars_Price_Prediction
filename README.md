# Cars Price Prediction Flask APP

Purpose; the user can learn how much she/he can sell own car.

### Dataset

Web scraping has been applied to create the dataset. Used **Selenium** library ([web_scraping.py](https://github.com/sevvalyogurtcuoglu/Cars_Price_Prediction/blob/master/web_scraping.py))
 
### Regression 
* While regression and prediction, modeling was done with ***Gradient Boosting Regressor and XGBoost Regressor***

* Hyperparameter tuning processing were performed to obtain the best performance results. ( [classification.py](https://github.com/sevvalyogurtcuoglu/Cars_Price_Prediction/blob/master/classification.py ))

* The model trained with Gradient Boosting Regressor for use in the Flask app was saved ( [train.py](https://github.com/sevvalyogurtcuoglu/Cars_Price_Prediction/blob/master/train.py ) ). Model registration was carried out with the joblib module ( [model.pkl](https://github.com/sevvalyogurtcuoglu/Cars_Price_Prediction/blob/master/model.pkl) ).

## Flask App

The flask app is in ([app.py](https://github.com/sevvalyogurtcuoglu/Cars_Price_Prediction/blob/master/app.py) ).

![Adsız3](https://user-images.githubusercontent.com/33968347/88243771-2f5ee700-cc9a-11ea-8d2c-caa2eccd95c8.png)

if you want to look at the screenshots in detail, [Click please](https://github.com/sevvalyogurtcuoglu/Cars_Price_Prediction/tree/master/screenshots)
