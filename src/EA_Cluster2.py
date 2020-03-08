#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:31:45 2020

@author: salim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz 
from xgboost import plot_tree
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import chisquare
from scipy import stats
from pandas import DataFrame


#os.chdir('/Users/salim/Downloads/examen_python/data')
#READ THE DATASETS
fifa19_numerical = pd.read_csv("../data/numeric_data.csv", index_col=0)
fifa19_categorical = pd.read_csv("../data/categorical_data.csv", index_col=0)

#MERGE THE DATASETS BY ID
fifa19 = fifa19_numerical.merge(fifa19_categorical, left_on='ID',right_on='ID')
fifa19 = fifa19.drop(['Name', 'Nationality', 'Club'], axis=1)
fifa19.columns



#CHECK THE DATASET FILEDS
fifa19.info
fifa19.info

#CHECK THE DATASET STATS 
fifa19.describe()
fifa19.describe()

#COMPROBATION OF OUR DATA
fifa19.head()
#Analisis of the null values
fifa19.isnull().sum()[fifa19.isnull().sum() > 0]



# Save table to varible X
X = fifa19
fifa19.columns
#fifa19 = fifa19.drop(columns=['Id'])


#SPLIT OUR DATA TO DO THE ONE HOT 
categorical_vars = set(X.columns[X.dtypes == object])
numerical_vars = set(X.columns) - categorical_vars
categorical_vars = list(categorical_vars)
numerical_vars = list(numerical_vars)

X[categorical_vars] = X[categorical_vars].astype(str)

print(categorical_vars)
print(numerical_vars)


#ONE HOT ENCODING

ohe = OneHotEncoder(sparse = False)
ohe_fit = ohe.fit(X[categorical_vars])
X_ohe = pd.DataFrame(ohe.fit_transform(X[categorical_vars]))
X_ohe.columns = pd.DataFrame(ohe_fit.get_feature_names())

X[categorical_vars].head()

X_ohe.head()

#JOIN THE DATASETS
X = pd.concat((X_ohe, X[numerical_vars].reset_index()), axis=1)

#X=X.drop(columns=['index'])
X=X.drop(columns=['ID'])


X.dtypes


#Using Pearson Correlation
fig=plt.figure(figsize=(18,16))
cor = fifa19_numerical.corr()
sns.heatmap(cor, vmin=-1, cmap='BrBG')
print("Show Correlation heatmap")
plt.show()
fig.savefig('correlation_matrix2.pdf')
#We have some negative correlations that are linked to the abilitys, position and skills of players like the players(defenders) who standing Tackle is Quality and is normal that the finishing is bad  
#Weight with SprintSpeed
#Weight with Reaction 
#Weight with ball control 

#Chi square correlation 

#scipy.stats.chisquare(fifa19_categorical["Work Rate"].value_counts())



#Normalization of the dataset 
sc = StandardScaler()
sc.fit(fifa19_numerical)
fifa_norm_numerical = sc.transform(fifa19_numerical)
plt.hist(fifa_norm_numerical)


#PCA NORMALIZED

pca = PCA().fit(fifa_norm_numerical)


#Graph of the variance la suma acumulativa de la varianza explicada()
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #para cada componente
plt.title('FIFA 19 Dataset Explained Variance')
print("Check if the PCA is worth using")
plt.show()



#ELBOW METHOD
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(fifa_norm_numerical)
    distortions.append(kmeanModel.inertia_)


# Let´s plot our result    
plt.figure(figsize=(8,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('The Elbow Method showing the optimal k')
print("Check the best value of K")
plt.show()


#K PROTOTYPE

kmeanModel = KMeans(n_clusters=4, init='k-means++', random_state=0, max_iter=500)
kmeanModel.fit(fifa_norm_numerical)
y_kmeans = kmeanModel.predict(fifa_norm_numerical)
centers = kmeanModel.cluster_centers_

kmeans_df = pd.DataFrame(centers)
fifa_norm_numerical = pd.DataFrame(fifa_norm_numerical)

kmeans_df.columns = fifa_norm_numerical.columns



y=kmeanModel.labels_



#Cluster density (FEATURE ARE BALANCED)
print("Cluster density label")
plt.hist(kmeanModel.labels_)
plt.show()

#CREATE OUR CLUSTERS
print(np.unique(kmeanModel.labels_, return_counts=True))

#y[y ==0] = 'PLAYER'
#y[y ==1] = 'GK'
y1= y.astype('object')
y1[y1 ==3] = 'STRIKER'
y1[y1 ==1] = 'MIDELFILDER'
y1[y1 ==2] = 'GOAL_KEEPER'
y1[y1 ==0] = 'DEFENDER'

print(y1)

################## XGBOOST ##############
#SPLIT THE DATA BETWEEN TRAIN AND TEST

X_train, X_test, y1_train, y1_test = train_test_split(fifa19_numerical, y1, test_size=0.20)
print(X_train.shape)
print(X_test.shape)
print(y1_train.shape)
print(y1_test.shape)


#CROSS VALIDATION
def cross_val(X_train, y1_train, model):
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = model, X = X_train, y = y1_train, cv = 5)
    return accuracies.mean()


# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y1_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_cross = cross_val(X_train, y1_train, model)
    
    # Return the performance metric
    return model_cross


#  RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
random_cross = fit_and_evaluate(random)
y_predrf = random.predict(X_test)
print('Random Forest Performance on the test set: Cross Validation Score = %0.4f' % random_cross)

# XGBClassifier

gb = XGBClassifier()
gb_cross = fit_and_evaluate(gb)
y_predxgb = gb.predict(X_test)
print('Gradiente Boosting Classification Performance on the test set: Cross Validation Score = %0.4f' % gb_cross)



pd.options.display.float_format = '{:.3f}'.format
# Extract the feature importances into a dataframe
feature_results = pd.DataFrame({'feature': list(fifa19_numerical.columns), 
                                'importance': random.feature_importances_})

# Show 
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)

feature_results.head(53)
model = XGBClassifier()

model.fit(fifa19_numerical, y1)

N= plot_tree(model)
plt.show(N)
plt.savefig("tree3.png")




#DECISION TREE
model_cart = DecisionTreeClassifier(max_depth=3)
model_cart.fit(fifa19_numerical, y1)
model_cart.classes_
model_cart.score(fifa19_numerical, y1)
KPI = list (fifa19_numerical.columns.values)  
dot_data = tree.export_graphviz(model_cart, feature_names = KPI, class_names=model_cart.classes_, filled=True, rounded=True, out_file=None)
graph = graphviz.Source(dot_data)
graph
######3
#SELECTION OF FEATURES 
fifa_KPI = ['Overall','Reactions','Crossing','Special','Finishing','Positioning','Dribbling']
fifa19_reduced = fifa19_numerical

for kpi in fifa_KPI:
    fifa19_reduced = fifa19_reduced[fifa19_reduced[kpi].isna() == False]
    
X1 = np.array(fifa19_reduced[fifa_KPI].values)
X_reduced =pd.DataFrame(X1, columns = fifa_KPI)

print("Scatter matrix features vs features clustered")
scatter_matrix(X_reduced, c=y,  alpha = 0.9,  figsize = (15, 15), diagonal = 'kde')


#####################################OTHER CLUSTERIZATION###########################

#K PROTOTYPE

kmeanModel2 = KMeans(n_clusters=4, init='k-means++', random_state=0, max_iter=500)
kmeanModel2.fit(X_reduced)
y_kmeans2 = kmeanModel2.predict(X_reduced)
centers2 = kmeanModel2.cluster_centers_

kmeans_df2 = pd.DataFrame(centers2)

kmeans_df2.columns = X_reduced.columns



y2=kmeanModel2.labels_
print("Print the new scatter matrix")
scatter_matrix(X_reduced, c=y2,  alpha = 0.9,  figsize = (15, 15), diagonal = 'kde')

y3= y2.astype('object')
y3[y3 ==3] = 'BRONZE'
y3[y3 ==1] = 'SILVER'
y3[y3 ==2] = 'GOLD'
y3[y3 ==0] = 'SPECIAL'
print("The new cluster density")
plt.hist(kmeanModel2.labels_)
plt.show()


#DECISION TREE
model_cart = DecisionTreeClassifier(max_depth=3)
model_cart.fit(X_reduced, y3)
model_cart.classes_
model_cart.score(X_reduced, y3)
KPI = list(X_reduced.columns.values)  

# dot_data = tree.export_graphviz(model_cart, feature_names = KPI, class_names=model_cart.classes_, filled=True, rounded=True, out_file=None)
# graph = graphviz.Source(dot_data)
# graph


print("The scatter matrix based on the new features")




plt.scatter(X1[:, 1], X1[:, 4],  c=y, cmap='rainbow', s=100)
plt.xlabel("Overall")
plt.ylabel("Special")





