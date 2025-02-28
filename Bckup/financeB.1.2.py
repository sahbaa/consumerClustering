
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, recall_score, precision_score, accuracy_score

def ProcessingAndPlotting(InData,kind='debt'):

    
    # Data Cleaning
    # ===================================================
    # Scale The Data
    def scaler (xtr):
        scaler = MinMaxScaler()
        xtr = scaler.fit_transform(xtr)
        xresult = pd.DataFrame(xtr,columns=scaler.get_feature_names_out())
        return xresult
    
    # Data For Plotting
    RealDf = InData.copy(deep=True)

    # Data Preprocessing
    # ===================================================
    
    InData = scaler(InData)

    # Training 
    # ===================================================
    # Hierarchical Clustering For Finding Centroids
    hirechical_clustering = AgglomerativeClustering(n_clusters=3,linkage='ward')
    hirechical_labels= hirechical_clustering.fit_predict(InData)
    hirechical_centroids = np.array([InData[hirechical_labels==i].mean(axis=0) for i in range(3)])

    # KMeans Clustering
    k_mean_model = KMeans(n_clusters=3,init=hirechical_centroids,max_iter=400,algorithm='elkan')
    k_mean_model.fit(InData)
    labels = k_mean_model.predict(InData)
    # Save The Centroids
    centroids = k_mean_model.cluster_centers_
    np.savetxt(f'Centroids_{kind}.csv',centroids,delimiter=',')

    # Number Of Each Cluster

    # Evaluation
    sil_avg = silhouette_score(InData,labels)

    # Definign Target For Both DataFrames
    InData['Target'] = labels
    RealDf['Target'] = labels
    # Storing Targetted The Data

    # Dfine Columns To Plot
    columns_to_plot = [col for col in RealDf.columns if col not in ['Target']]

    # Computing The Average Of Each Feature For Each Cluster
    cluster_means = RealDf.groupby("Target")[columns_to_plot].mean()

    # Plotting The Grouped Bar Chart
    # ===================================================

    cluster_means.T.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    plt.xlabel("Financial Features")
    plt.ylabel("Average Value")
    plt.title(f"Comparison of {kind} Behavior Across Clusters")
    plt.legend(title="Cluster")
    plt.xticks(rotation=45)
    png_filename = os.path.join('.', f'BrGraph_{kind}.png')
    # plt.show()
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()



    # Plotting Second Graph
    # ===================================================
    for _, row in RealDf.iterrows():
        plt.plot(["Tir", "Mordad", "Shahrivar"], 
                [row[f"{kind}_tir"], row[f"{kind}_mordad"], row[f"{kind}_shahrivar"]], 
                marker='o', linestyle='-', alpha=0.5)

    # Adding labels and title
    plt.xlabel("Month")
    plt.ylabel(f"{kind} Amount")
    plt.title(f"{kind} Trend Over Three Months for Each Customer")
    plt.grid(True)
    png_filename = os.path.join('.', f'Clusters_{kind}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()



    # Plotting The Line Chart
    # ===================================================
    plt.figure(figsize=(10, 6))
    for cluster in cluster_means.index:
        plt.plot(["Tir", "Mordad", "Shahrivar"], cluster_means.loc[cluster], marker='o', linestyle='-', label=f"Cluster {cluster}")

    # Adding labels and title
    plt.xlabel("Month")
    plt.ylabel("Average Debt Amount")
    plt.title(f"{kind} Trend Over Three Months for Each Cluster")
    plt.legend(title="Cluster")
    plt.grid(True)
    png_filename = os.path.join('.', f'debtClusters2_{kind}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()

    # Calculate Number Of Each Cluster in InData
    cls,numbers = np.unique(InData['Target'],return_counts=True)
    N_of_Cl = pd.DataFrame(numbers,index=cls)
    np.savetxt(f'ClusterNumber_comulative_{kind}.csv',N_of_Cl,delimiter=',')
    return InData

# Data Loading
# ===================================================
# Tir
InData_Tir          = pd.read_csv('Tir\\result_Tir.csv')

# Mordad
InData_Mordad       = pd.read_csv('Mordad\\result_Mordad.csv')

# Shahrivar
InData_Shahrivar    = pd.read_csv('Shahrivar\\result_Shahrivar.csv')

# Merging Data On Row to get the finance Behaviour
# ========================
df12        = InData_Tir.merge(InData_Mordad, how='inner', on='Code')
df123       = df12.merge(InData_Shahrivar, on='Code', how='inner')
tempData    = df123.copy(deep=True)

# Make Data Ready for Analysis
# Keep Only Columns With Debt
InData              = tempData.drop(tempData.columns[[0,1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15]], axis=1)
ConData             = tempData.drop(tempData.columns[[0,1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15]], axis=1)
# Finding Mode Target Between 3 Months

InData.columns=['debt_tir',
              'debt_mordad',
              'debt_shahrivar',
              ]

ConData.columns=['Con_tir','Con_mordad','Con_shahrivar']

InData = ProcessingAndPlotting(InData)
ConData = ProcessingAndPlotting(ConData,'Con')
# ===================================================

# ===================================================





# ===================================================
# Training Tree Model
# ===================================================
DataTree = df123.copy(deep=True)
# Drop Unwanted Columns
DataTree = DataTree.drop(DataTree.columns[[0,5,6,10,11,15]], axis=1)
# Adding Target
DataTree['Target'] = InData['Target']
# Definning Columns Name

DataTree.columns = ['CustomerCode', 'total_tir', 'debt_tir', 'avgConsuming_tir', 'total_mordad',
       'debt_mordad', 'avgConsuming_mordad', 'total_shahrivar', 'debt_shahrivar', 'avgConsuming_shahrivar', 'Target']

# Indexing Data
DataTree = DataTree.set_index('CustomerCode')       
# Defining Inputs and Outputs
inputs = DataTree.iloc[:,:-1]
outputs = DataTree.iloc[:,-1]
# Split Data
xtrain, xtest , ytrain, ytest = train_test_split(inputs,outputs,test_size=0.3,random_state=42)
#Over Sampling
smote = SMOTE(random_state=12,k_neighbors=1) 
xtrain_smote, ytrain_smote = smote.fit_resample(xtrain, ytrain)


# Find Best Model
def f1_scoring(y_true,y_pred):
    return f1_score(y_true, y_pred, average='macro')

gscvScored = make_scorer(f1_scoring)
model = DecisionTreeClassifier(random_state=42)
params = {'criterion' : ["gini", "entropy", "log_loss"],'max_depth' : [3,4,5,6],
          'class_weight' : [{0: 1.0, 1: 1.5, 2:10},{0: 1.0, 1: 1, 2: 1}]
          }
gscv = GridSearchCV(estimator=model,cv=10,scoring=gscvScored,param_grid=params)
best_model = gscv.fit(xtrain_smote,ytrain_smote)

#Predcition
y_pred = best_model.best_estimator_.predict(xtest)

# Evaluation
cnf = confusion_matrix(ytest,y_pred).T
f_score = f1_score(ytest,y_pred,average='macro')
print('confusionMatrix:','\n',cnf)
result = {'f_score':f_score,'recall':recall_score(ytest,y_pred,average='macro'),
          'percision': precision_score(ytest,y_pred,average='macro'),'accuracy':accuracy_score(ytest,y_pred)}
print(result)


# Plotting Tree
# ===================================================
plt.figure(figsize=(15, 10))
plot_tree(best_model.best_estimator_, feature_names=xtrain.columns, class_names=[str(cls) for cls in best_model.best_estimator_.classes_], filled=True)
png_filename = os.path.join('./', f'treeRule.png')
plt.savefig(png_filename, dpi=300, bbox_inches='tight')
plt.close()
plt.show()




# ExtractRules
tree_rules = export_text(best_model.best_estimator_, feature_names=list(xtrain.columns))

# Plotting Rules
print(tree_rules)

with open("./decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)
