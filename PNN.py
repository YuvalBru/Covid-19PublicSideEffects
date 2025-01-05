from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np


data = pd.read_csv("C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/CatBoost Model Trial - Boosting/data_adv_ml_top.csv")

cat_col = ['תאריך חיסון- שנה', 'סוג מרכז רפואי','קבוצת גיל','דיווח עבור']

#Data preparation for neural network
data['תאריך חיסון- מספר שבוע בשנה'] = pd.to_numeric(data['תאריך חיסון- מספר שבוע בשנה'], errors='coerce')
median_value = data['תאריך חיסון- מספר שבוע בשנה'].median()
data['תאריך חיסון- מספר שבוע בשנה'].fillna(median_value, inplace=True)
data['שבוע הריון'] = pd.to_numeric(data['שבוע הריון'], errors='coerce')
data['שבוע הריון'] = data['שבוע הריון'].fillna(0)
data['מספר מנה'] = data['מספר מנה'].fillna(data['מספר מנה'].median())
data[cat_col] = data[cat_col].fillna('Unknown')
data['פנה לטיפול רפואי'] = data['פנה לטיפול רפואי'].map({'כן': 1, 'לא': 0}).fillna(0)
data = data[data['מין'].notna()]
data['חיסון ניתן בהריון'] = data['חיסון ניתן בהריון'].fillna(0)
data['נטילת תרופה'] = data['נטילת תרופה'].fillna(0)
data['אישפוז'] = data['אישפוז'].fillna(0)

data = pd.get_dummies(data, columns= cat_col, drop_first = True)

#data.to_csv("C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/data/NN_data.csv", encoding="utf-8-sig")
x = data.drop('MEDdra PT manually extracted from free text', axis=1)
y = data['MEDdra PT manually extracted from free text']
#Splitting data for training testing and validation during training
x_temp, x_train, y_temp, y_train = train_test_split(x,y, test_size = 0.7, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp,y_temp, test_size=0.5,random_state=42)

#Defining our architecture after many tests
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

#Defining the loss relevant metric and optimizer algorithm
model.compile(optimizer=Adam(learning_rate=0.00062), loss='categorical_crossentropy', metrics=['accuracy'])
#Defining early stopping in case of over fitting or model no longer improving
early_stopping = EarlyStopping(monitor = 'val_loss',patience=35, restore_best_weights = True)

#Attempt to tackle the problem of class imbalance using weights however the attempt failed and we achieved using this method lower accuracy
weight_no_diagnosis = len(x_train)/(5147*2)
weight_diagnosis = (len(x_train)/(1106*2))

encoder = OneHotEncoder(sparse_output=False)

y_train_one_hot = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_val_one_hot = encoder.fit_transform(y_val.values.reshape(-1,1))
y_test_one_hot = encoder.fit_transform(y_test.values.reshape(-1,1))
#Model training
model_keeper = model.fit(x_train,y_train_one_hot, validation_data=(x_val,y_val_one_hot), epochs = 300,
                         batch_size =128,
                         #class_weight ={0:weight_no_diagnosis, 1: weight_diagnosis},
                         callbacks = [early_stopping] )

test_loss, test_accuracy = model.evaluate(x_test,y_test_one_hot)
print(f'Accuracy is: {test_accuracy}')
train_loss = model_keeper.history['loss']
val_loss = model_keeper.history['val_loss']
#Plotting loss convergence graph
plt.plot(train_loss, label='Training Loss', color= 'red')
plt.plot(val_loss, label= 'Validation Loss', color= 'blue')
plt.title('Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best', title='Legend', fontsize=10)

plt.show()
#Confusion matrix and metrics calculations (aside from accuracy which was calculated a few lines before)
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)


if len(y_test.shape) > 1:
    y_test = np.argmax(y_test, axis=1)

precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
print(f'Recall: {recall}, Precision: {precision}')
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

plt.show()

model_keeper.save_model('C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/Customized Neural Network Trial/PNN_model.zip')
