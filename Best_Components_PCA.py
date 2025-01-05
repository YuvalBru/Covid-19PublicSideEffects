import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/yuval/PycharmProjects/Advanced_Topics_In_MachineLearning/data/data_adv_ml_top.csv")

data = data.drop(columns=["Unnamed: 0"])
data['פנה לטיפול רפואי'] = data['פנה לטיפול רפואי'].map({'כן': 1, 'לא': 0})
cat_list = ['תאריך חיסון- שנה', 'סוג מרכז רפואי','קבוצת גיל','דיווח עבור']

num_list = []


for item in data.columns:
    if item not in cat_list:
        num_list.append(item)


data['תאריך חיסון- מספר שבוע בשנה'] = pd.to_numeric(data['תאריך חיסון- מספר שבוע בשנה'], errors='coerce')
median_value = data['תאריך חיסון- מספר שבוע בשנה'].median()
data['תאריך חיסון- מספר שבוע בשנה'].fillna(median_value, inplace=True)
data['שבוע הריון'] = pd.to_numeric(data['שבוע הריון'], errors='coerce')
data[num_list] = data[num_list].fillna(0)

cat_transformer = OneHotEncoder(handle_unknown='ignore')

#Creating pre processing before feeding the data to the model
preprocessor = ColumnTransformer(
    transformers=[('cat', cat_transformer, cat_list),
                  ('num', StandardScaler(), num_list)
                  ]
)

pipeline_pca = make_pipeline(preprocessor, PCA(n_components=10))
explained_variances = []

#Collecting data in order to plot the Elbow Method
for n in range(1,40):
    pca = PCA(n_components=n)
    pipeline = make_pipeline(preprocessor, pca)
    pipeline.fit(data)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

#Plotting the Elbow Method
plt.plot(range(1,40), explained_variances, marker='o', linestyle='--', color='b')
plt.title('Elbow Method: PCA Explained Variance')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Explained Variance')
plt.legend()
plt.grid()
plt.show()

#Executing the optimal PCA for our goals
pipeline_pca = make_pipeline(preprocessor, PCA(n_components=30)) # 10 components for 50-60 percentage variance and 30 components for 90% percentage variance
pca_result = pipeline_pca.fit_transform(data)



pca = pipeline_pca.named_steps['pca']
components = pca.components_


encoder = preprocessor.named_transformers_['cat']
feature_names = encoder.get_feature_names_out(cat_list)

cat_feature_names = encoder.get_feature_names_out(cat_list)

all_feature_names = list(cat_feature_names) + num_list
#Extracting top5 most influential features from each component
for i, component in enumerate(components):
    print(f"\nPrincipal Component {i+1}:")
    sorted_idx = np.argsort(np.abs(component))[::-1]
    max_features = min(5, len(all_feature_names))
    for idx in sorted_idx[:max_features]:
        print(f"  {all_feature_names[idx]}: {component[idx]}")


