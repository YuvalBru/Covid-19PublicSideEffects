import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import re


data = pd.read_csv("C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/CatBoost Model Trial - Boosting/data_adv_ml_top.csv")
def clean_column_names(columns): #ChatGPT
    return [re.sub(r'[^\w\s]', '_', col) for col in columns]

#End ChatGPT
data.columns = clean_column_names(data.columns)

binary_features = ['חיסון ניתן בהריון','פנה לטיפול','אישפוז','נטילת תרופה','מין']
categorical_features =[]
#Sorting which columns are categorical
for col in data.columns:
    if (col not in binary_features) and ('fval' not in col)  and ('שבוע הריון' not in col) and ( 'מספר שבועות חיסון' not in col) and ('מספר מנה' not in col) and ('MEDdra' not in col):
        data[col] = data[col].astype('category')
        categorical_features.append(col)


x = data[[col for col in data.columns if col !='MEDdra PT manually extracted from free text']]
y = data['MEDdra PT manually extracted from free text']

#Splitting for training
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.3, random_state=42)

#Preparing for training in needed format
train_data = lgb.Dataset(x_train,label = y_train, categorical_feature= categorical_features)
test_data = lgb.Dataset(x_test,label = y_test, categorical_feature=categorical_features,reference=train_data)

#Defining model parameters
model_parameters = {
        'objective':'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'num_leaves': 127,
        'max_depth': -1,
        'learning_rate': 0.1,
        'feature_fraction': 1
    }
#Training model
lgb_model = lgb.train(
    params= model_parameters,
    train_set=train_data,
)

y_pred = lgb_model.predict(x_test)
y_pred_binary =[1 if x > 0.5 else 0 for x in y_pred]
#Calculating metrics
accuracy = accuracy_score(y_test,y_pred_binary)
precision = precision_score(y_test,y_pred_binary)
recall = recall_score(y_test,y_pred_binary)
print(f"Accuracy is: {accuracy} \n precision is: {precision} \n recall is: {recall}")



conf_matrix = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

plt.show()
#Extracting most important features in the model
feature_importances = lgb_model.feature_importance()

feature_importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)
