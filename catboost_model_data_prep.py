import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("C:/Users/yuval/Desktop/vacseffectpublic30.10.22.csv")
base_q1 = 'מועד התופעה'
base_q2 = 'משך זמן התופעה'
base_q3 = 'תגובות דומות בעבר'
base_q4 = 'חיסור בעבודה'
base_q5 = 'בעיות בתפקוד היומיומי'
f = 0
weights = [0.4, 0.9, 0.9, 0.1, 0.7]
binary_features = ['חיסון ניתן בהריון','פנה לטיפול','אישפוז','נטילת תרופה','מין']
category_features =[]
scaler = StandardScaler()

for item in data.columns:
    if (base_q1 in item) or (base_q2 in item) or (base_q3 in item) or (base_q4 in item) or (base_q5 in item):
        data[item] = data[item].astype('category').cat.codes + 1 #ChatGPT
        data[item] = data[item].astype('int')

# We create linear mapping f: R^(5) -> R^(1) where we attempt to minimize the amount of features in the dataset.
# We basically take the asumption that we can keep the same amount of "information" of those 6 features in one feature
# and therefore we use this technique to lower the amount of resources needed for training the model.
# It should be taken into account that some categories are more dominant than others in the dataset therefore when we calculate the values
# for the f function normalization is needed.

for index, column in enumerate(data.columns):
    if 11 <= index <= 193:
        if index % 6 == 5:
            data[column] = data[column].astype(int)
            main_question = column
            follow_up_columns = data.columns[index + 1:index + 6]

            f_values = []
            for i, row in data.iterrows():
                f = 0
                for w, follow_up_col in zip(weights, follow_up_columns):
                    if not isinstance(row[follow_up_col], str) and not isinstance(data[follow_up_col].sum(), str):
                      val = row[follow_up_col] / data[follow_up_col].sum()
                      f += w * val
                f *= row[main_question]
                f_values.append(f)
            data[f'{main_question}_fval'] = f_values

columns_to_drop = []

for index, column in enumerate(data.columns):
    if 11 <= index <= 193:
        if index % 6 == 5:
            follow_up_columns = data.columns[index:index + 6]
            columns_to_drop.extend(follow_up_columns)

columns_to_drop.extend(data.columns[0:3])

data.drop(columns=columns_to_drop, inplace=True)

for col in data.columns:
    if (col in binary_features) and (col != 'מין'):
        data[col] = data[col].map({'כן': 1, 'לא': 0})
    if col == 'מין':
        data[col] = data[col].map({ 'זכר':1 , 'נקבה': 0})
    if (col not in binary_features) and ('fval' not in col)  and ('שבוע הריון' not in col) and ( 'מספר שבועות חיסון' not in col) and ('מספר מנה' not in col) and ('MEDdra' not in col):
        category_features.append(col)
        data[col].fillna('NaN', inplace = True)
        data[col] = data[col].astype('category')

data['MEDdra PT manually extracted from free text'].fillna(0, inplace=True)

data['MEDdra PT manually extracted from free text'] = data['MEDdra PT manually extracted from free text'].apply(lambda x: 1 if x != 0 else 0)  #ChatGPT

fval_columns = [col for col in data.columns if 'fval' in col]

data[fval_columns] = scaler.fit_transform(data[fval_columns])

#data.to_csv('./data_adv_ml_top.csv', encoding="utf-8-sig")

x = data[[col for col in data.columns if col != 'MEDdra PT manually extracted from free text']]
y = data['MEDdra PT manually extracted from free text']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state= 42)


catboost_model  = CatBoostClassifier(iterations=800, depth=10, learning_rate=0.015, loss_function='Logloss', l2_leaf_reg=10)
catboost_model.fit(x_train,y_train , cat_features=category_features)

y_pred = catboost_model.predict(x_test)

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
feature_importances = catboost_model.get_feature_importance() #ChatGPT

feature_importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)
print(f'The accuracy is: {accuracy} \n The precision of the model is {precision} \n The reacll of the model is {recall}')
