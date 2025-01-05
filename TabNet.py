import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier

data = pd.read_csv('C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/data/NN_data.csv')

x = data.drop('MEDdra PT manually extracted from free text', axis=1)
y = data['MEDdra PT manually extracted from free text']

x = x.to_numpy()
y = y.to_numpy().flatten()
x = x.astype(np.float32)

x_temp, x_train, y_temp, y_train = train_test_split(x, y, test_size=0.7, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

param_grid = {
    'n_d': [8, 16, 32],
    'n_a': [8, 16, 32],
    'learning_rate': [0.001, 0.005],
    'batch_size': [128, 256],
    'max_epochs': [50, 100, 200]
}

def grid_search_tabnet(param_grid, X_train, y_train):
    best_params = None
    best_score = -np.inf

    for n_d in param_grid['n_d']:
        for n_a in param_grid['n_a']:
            for lr in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['max_epochs']:
                        model = TabNetClassifier(
                            n_d=n_d,
                            n_a=n_a,
                            optimizer_params=dict(lr=lr)
                        )
                        model.fit(
                            X_train=X_train, y_train=y_train,
                            eval_set=[(x_val, y_val)],
                            eval_metric=['accuracy'],
                            max_epochs=epochs,
                            batch_size=batch_size,
                            patience=20,
                        )
                        score = model.best_cost
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'n_d': n_d,
                                'n_a': n_a,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'max_epochs': epochs
                            }

    return best_params, best_score

#best_params, best_score = grid_search_tabnet(param_grid, x_train, y_train)
#print(f"Best Params: {best_params}, Best Score: {best_score}")

#best_params
final_model = TabNetClassifier(
    n_d= 8 ,
    #best_params['n_d'],
    n_a= 8,
    #best_params['n_a'],
    optimizer_params=dict(lr=0.005),
                          #best_params['learning_rate'])
)

final_model.fit(
    X_train=x_train, y_train=y_train,
    eval_set=[(x_val, y_val)],
    eval_metric=['accuracy'],
    max_epochs=50,
    #best_params.get('max_epochs', 50),
    batch_size=256,
    #best_params['batch_size'],
    patience=20,
)

y_pred = final_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Test, Accuracy: {accuracy}, Precision {precision}, Recall {recall} ")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

final_model.save_model('C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/TabNet Model Trial - Neural Network/tabnet_model.zip')
