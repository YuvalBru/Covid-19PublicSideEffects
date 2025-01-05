import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/CatBoost Model Trial - Boosting/data_adv_ml_top.csv')
binary_features = ['חיסון ניתן בהריון','פנה לטיפול','אישפוז','נטילת תרופה','מין']

for col in data.columns:
    if (col not in binary_features) and ('fval' not in col)  and ('שבוע הריון' not in col) and ( 'מספר שבועות חיסון' not in col) and ('מספר מנה' not in col) and ('MEDdra' not in col):
        data[col] = data[col].astype('category').cat.codes + 1

x = data[[col for col in data.columns if col != 'MEDdra PT manually extracted from free text']]
y = data['MEDdra PT manually extracted from free text']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state= 42)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', max_depth = 10, learning_rate = 0.01,n_estimators = 500, colsample_bytree = 0.8, subsample =0.8, reg_lambda = 10)

xgb_model.fit(x_train,y_train, verbose = True)

y_pred = xgb_model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

precision =  precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

plt.show()

feature_importances = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)
print(f'The accuracy is: {accuracy} \n The precision of the model is {precision} \n The reacll of the model is {recall}')
