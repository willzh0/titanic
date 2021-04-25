# titanic

### Overview:

This is my code for the titanic problem on Kaggle, using scikit-learn. It consists of two files:
- train.py, which preprocesses the data, trains several different machine learning models, and saves the model with the best accuracy score (using cross-validation)
with joblib. Training may take several hours on a regular PC. 
- test.py, which will run the saved model on train.csv, and produces predictions.csv, which are the model's predictions of who will survive and die. 

After a bit of testing, the best model seems to either use gradient boosting or a random forest and achieves around 77-78% accuracy. With 
some more feature engineering and better model selection, this accuracy could likely be improved. 
