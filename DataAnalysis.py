# Imports
#======================================================
import os
import jdatetime
from matplotlib import pyplot as plt
import numpy as np
import pandas               as pd
import seaborn              as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler 


# Functions
#======================================================
def outlier_detection(xtr):
    xtemp = xtr.copy(deep=True)

    # Imputation
    for c in xtr.columns:
        imputer     = SimpleImputer(strategy='median')
        xtemp[[c]]  = imputer.fit_transform(xtemp[[c]])

    # Scaling
    st_scaler                       = MinMaxScaler()
    xtemp                           = st_scaler.fit_transform(xtemp)
    data                            = pd.DataFrame(xtemp,columns=st_scaler.get_feature_names_out(),index=xtr.index)
  
    # Outlier Detection
    model                           = IsolationForest(contamination=0.1 ,n_estimators=1000)
    out                             = model.fit_predict(data)

    data['outs']                    = out
    out_indx                        = pd.Index(data[data['outs']==-1].index)
  
    return out_indx


def feature_screening(xtr):
    cv_threshold                = 0.1
    cv_columns                  = xtr.std()/xtr.mean()
    del_cols                    = cv_columns[cv_columns<cv_threshold].index
    xtr                         = xtr.drop(del_cols,axis=1)
    return xtr



def scaler (xtr):
    scaler = MinMaxScaler()
    xtr = pd.DataFrame(scaler.fit_transform(xtr),columns=scaler.get_feature_names_out())
    return xtr




# Data Loading
#======================================================
InData_Tir                  = pd.read_csv("Tir.csv")
InData_Mordad               = pd.read_csv("Mordad.csv")
InData_Shahrivar            = pd.read_csv("Shahrivar.csv")


# Indexing
#======================================================
# Tir
InData_Tir_Idx              = InData_Tir.set_index('Row')
InData_Tir_Idx['Debt'] = InData_Tir_Idx['Debt'].apply(lambda x :0 if x<=999 else x)
InData_Tir_Idx['MainCost'] = InData_Tir_Idx['MainCost'].apply(lambda x: 0 if x<0 else x)
deleting_indx = InData_Tir_Idx[InData_Tir_Idx['State']!='Ok'].index
InData_Tir_Idx = InData_Tir_Idx.drop(InData_Tir_Idx.index.intersection(deleting_indx), axis=0)
InData_Tir_Idx = InData_Tir_Idx.drop(['State','ReadNumber'],axis=1)

# Mordad
InData_Mordad_Idx           = InData_Mordad.set_index('Row')
InData_Mordad_Idx['Debt'] = InData_Mordad_Idx['Debt'].apply(lambda x :0 if x<=999 else x)
InData_Mordad_Idx['MainCost'] = InData_Mordad_Idx['MainCost'].apply(lambda x: 0 if x<0 else x)
deleting_indx = InData_Mordad_Idx[InData_Mordad_Idx['State']!='Ok'].index
InData_Mordad_Idx = InData_Mordad_Idx.drop(InData_Mordad_Idx.index.intersection(deleting_indx), axis=0)
InData_Mordad_Idx = InData_Mordad_Idx.drop(['State','ReadNumber'],axis=1)

# Shahrivar
InData_Shahrivar_Idx        = InData_Shahrivar.set_index('Row')
InData_Shahrivar_Idx['Debt'] = InData_Shahrivar_Idx['Debt'].apply(lambda x :0 if x<=999 else x)
InData_Shahrivar_Idx['MainCost'] = InData_Shahrivar_Idx['MainCost'].apply(lambda x: 0 if x<0 else x)
deleting_indx = InData_Shahrivar_Idx[InData_Shahrivar_Idx['State']!='Ok'].index
InData_Shahrivar_Idx = InData_Shahrivar_Idx.drop(InData_Shahrivar_Idx.index.intersection(deleting_indx), axis=0)
InData_Shahrivar_Idx = InData_Shahrivar_Idx.drop(['State','ReadNumber'],axis=1)

# Preprocessing
#======================================================
# Convert "ReadDate" from Jalaali to Gregorian
# Tir
InData_Tir_Idx['ReadDate'] = InData_Tir_Idx['ReadDate'].apply(lambda x: jdatetime.datetime.strptime(x, '%Y/%m/%d').togregorian().strftime('%Y/%m/%d'))

# Mordad
InData_Mordad_Idx['ReadDate'] = InData_Mordad_Idx['ReadDate'].apply(lambda x: jdatetime.datetime.strptime(x, '%Y/%m/%d').togregorian().strftime('%Y/%m/%d'))

# Shahrivar
InData_Shahrivar_Idx['ReadDate'] = InData_Shahrivar_Idx['ReadDate'].apply(lambda x: jdatetime.datetime.strptime(x, '%Y/%m/%d').togregorian().strftime('%Y/%m/%d'))

# ================================================
# Remove Duplicate "Code" Columns by latest "ReadDate"
# Tir
InData_Tir_Idx = InData_Tir_Idx.sort_values('ReadDate').drop_duplicates('Code',keep='last')

# Mordad
InData_Mordad_Idx = InData_Mordad_Idx.sort_values('ReadDate').drop_duplicates('Code',keep='last')

# Shahrivar
InData_Shahrivar_Idx = InData_Shahrivar_Idx.sort_values('ReadDate').drop_duplicates('Code',keep='last')

# ================================================
# Drop "ReadDate" Column and "lastReadDate" column 
# Tir
InData_Tir_Idx = InData_Tir_Idx.drop(['ReadDate','LastReadDate','Consume'],axis=1)

# Mordad
InData_Mordad_Idx = InData_Mordad_Idx.drop(['ReadDate','LastReadDate','Consume'],axis=1)

# Shahrivar
InData_Shahrivar_Idx = InData_Shahrivar_Idx.drop(['ReadDate','LastReadDate','Consume'],axis=1)


# ================================================
# Feature Screening
# Tir
InData_Tir_Idx = feature_screening(InData_Tir_Idx)

# Mordad
InData_Mordad_Idx = feature_screening(InData_Mordad_Idx)

# Shahrivar
InData_Shahrivar_Idx = feature_screening(InData_Shahrivar_Idx)


# Outlier Detection
# Tir
out_indx_Tir = outlier_detection(InData_Tir_Idx)
InData_Tir_Idx  = InData_Tir_Idx.drop(out_indx_Tir,axis=0)
# Mordad
out_indx_Mordad = outlier_detection(InData_Mordad_Idx)
InData_Mordad_Idx  = InData_Mordad_Idx.drop(out_indx_Mordad,axis=0)
# Shahrivar
out_indx_Shahrivar = outlier_detection(InData_Shahrivar_Idx)
InData_Shahrivar_Idx  = InData_Shahrivar_Idx.drop(out_indx_Shahrivar,axis=0)

# Replace "unknown" with None
# Tir
InData_Tir_Idx = InData_Tir_Idx.replace({'unknown':None})

# Mordad
InData_Mordad_Idx = InData_Mordad_Idx.replace({'unknown':None})

# Shahrivar
InData_Shahrivar_Idx = InData_Shahrivar_Idx.replace({'unknown':None})


# Checking Missing Values
# Tir
MissingValues_Tir       = InData_Tir_Idx.isnull().sum()

# Mordad
MissingValues_Mordad    = InData_Mordad_Idx.isnull().sum()

# Shahrivar
MissingValues_Shahrivar = InData_Shahrivar_Idx.isnull().sum()

# Copy DataFrames For Plotting
# Tir
InData_Tir_Idx_Plot = InData_Tir_Idx.copy(deep=True)

# Mordad
InData_Mordad_Idx_Plot = InData_Mordad_Idx.copy(deep=True)

# Shahrivar
InData_Shahrivar_Idx_Plot = InData_Shahrivar_Idx.copy(deep=True)


# Scaling 
# Tir
InData_Tir_Idx = scaler(InData_Tir_Idx)

# Mordad
InData_Mordad_Idx = scaler(InData_Mordad_Idx)

# Shahrivar
InData_Shahrivar_Idx = scaler(InData_Shahrivar_Idx)


# Training 
#======================================================

# Finding Centroids Of Each Cluster
hirechical_clustering = AgglomerativeClustering(n_clusters=3,linkage='single')
# Centroids
hirechical_labels= hirechical_clustering.fit_predict(InData_Tir_Idx)
hirechical_centroids = np.array([InData_Tir_Idx[hirechical_labels==i].mean(axis=0) for i in range(3)])


# KMeans 
k_mean_model = KMeans(n_clusters=3,init=hirechical_centroids,n_init='auto',max_iter=400,algorithm='elkan',random_state=42)
k_mean_model.fit(InData_Tir_Idx)
labels_Tir = k_mean_model.predict(InData_Tir_Idx)
labels_Mordad = k_mean_model.predict(InData_Mordad_Idx)
labels_Shahrivar = k_mean_model.predict(InData_Shahrivar_Idx)
# Evaluation KMeans
sil_avg = silhouette_score(InData_Tir_Idx,labels_Tir)

# Add Target To DataFrames
# Tir
InData_Tir_Idx['Target'] = labels_Tir
InData_Tir_Idx_Plot['Target'] = labels_Tir

# Mordad
InData_Mordad_Idx['Target'] = labels_Mordad
InData_Mordad_Idx_Plot['Target'] = labels_Mordad

# Shahrivar
InData_Shahrivar_Idx['Target'] = labels_Shahrivar
InData_Shahrivar_Idx_Plot['Target'] = labels_Shahrivar




# Definign Mapping
cluster_names = {0: "High", 1: "Low", 2: "Moderate"}

# Mapping
# Tir
InData_Tir_Idx['Target'] = InData_Tir_Idx['Target'].map(cluster_names)
InData_Tir_Idx_Plot['Target'] = InData_Tir_Idx_Plot['Target'].map(cluster_names)

# Mordad
InData_Mordad_Idx['Target'] = InData_Mordad_Idx['Target'].map(cluster_names)
InData_Mordad_Idx_Plot['Target'] = InData_Mordad_Idx_Plot['Target'].map(cluster_names)

# Shahrivar
InData_Shahrivar_Idx['Target'] = InData_Shahrivar_Idx['Target'].map(cluster_names)
InData_Shahrivar_Idx_Plot['Target'] = InData_Shahrivar_Idx_Plot['Target'].map(cluster_names)

# Plotting
#======================================================
#First Plot
# Tir
ColumnsBar_Tir = InData_Tir_Idx_Plot.iloc[:, 1:-1].columns  # Get the names of the scores
for cluster in InData_Tir_Idx.Target.unique():
    # Set up the matplotlib figure with subplots for each score
    fig, axes = plt.subplots(1, len(ColumnsBar_Tir), figsize=(15, 4))
    fig.suptitle('Tir',fontsize=16, fontweight='bold')
    # Loop over each score and create a histogram plot
    for i, score in enumerate(ColumnsBar_Tir):
        ax = axes[i]  # Select the subplot axis
        ax.set_title(f'{score} Distribution\n(Cluster {cluster})')
        ax.set_xlabel(score)
        ax.ticklabel_format(style='plain', axis='both')
        
        if score=='MainCost':
            # bin_width = 500000  # Width of each bin (adjust as needed)
            # num_bins = int((4000000 - 0) / bin_width)
            ax.set_xlim([0, 2000000]) 
            ax.set_ylim([0, 10000]) 
            bin_edges = np.linspace(0, 6000000, 30 + 1)  
            ax.tick_params(axis='x', rotation=90) 
        elif score == 'Debt':
            # bin_width = 500000  # Width of each bin (adjust as needed)
            # num_bins = int((6000000 - 0) / bin_width)
            ax.set_xlim([0, 3000000])
            ax.set_ylim([0, 10000]) 
            bin_edges = np.linspace(0, 6000000, 30 + 1)
            ax.tick_params(axis='x', rotation=90)
        else:
            bin_edges=20
            ax.set_xlim([0, 100])
        ax.set_ylabel('Count')
        # Combine all data and cluster data into one DataFrame for plotting
        all_data = pd.DataFrame({score: InData_Tir_Idx_Plot[score], 'Type': 'All Data'})
        data_cluster = pd.DataFrame({score: InData_Tir_Idx_Plot[InData_Tir_Idx_Plot['Target'] == cluster][score], 'Type': f'Cluster {cluster}'})
        data_combined = pd.concat([all_data, data_cluster])

        # Plot histogram for each score
        sns.histplot(x=score, hue='Type', data=data_combined, ax=ax, 
                     palette=['pink', 'black'], bins=bin_edges, kde=False, stat='count', alpha=0.7)

        # Set the title and labels for each subplot
    plt.tight_layout()  # Adjust layout for better visualization
    png_filename = os.path.join('./Tir', f'cluster_{cluster}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory
    plt.show()  # Show the figure for the current cluster


# Second Plot 
# Tir
ColumnsBar_Tir_Sec = InData_Tir_Idx_Plot.iloc[:,1:-1].columns
for cl in InData_Tir_Idx_Plot['Target'].unique():
    df_cluster = InData_Tir_Idx_Plot[InData_Tir_Idx_Plot['Target']==cl]
    figu, axes2 = plt.subplots(1, len(ColumnsBar_Tir_Sec), figsize=(15, 4))
    figu.suptitle('Tir',fontsize=16, fontweight='bold')
    for i,sc in enumerate(ColumnsBar_Tir_Sec):
        ax2=axes2[i]
        ax2.ticklabel_format(style='plain', axis='both')
        if sc=='MainCost':
            # bin_width = 500000  # Width of each bin (adjust as needed)
            # num_bins = int((4000000 - 0) / bin_width)
            ax2.set_xlim([0, 2000000]) 
            ax2.set_ylim([0, 10000]) 
            bin_edges = np.linspace(0, 6000000, 30 + 1)  
            ax2.tick_params(axis='x', rotation=90) 
        elif sc == 'Debt':
            # bin_width = 500000  # Width of each bin (adjust as needed)
            # num_bins = int((6000000 - 0) / bin_width)
            ax2.set_xlim([0, 3000000])
            ax2.set_ylim([0, 10000]) 
            bin_edges = np.linspace(0, 6000000, 30 + 1)
            ax2.tick_params(axis='x', rotation=90)
        else:
            bin_edges=20
            ax2.set_xlim([0, 100])
        sns.histplot(df_cluster[sc], bins=bin_edges, kde=True, color='blue', edgecolor='black', alpha=0.7,ax=ax2)
        
        ax2.set_title(f'{sc} Distribution\n(Cluster {cl})')
        ax2.set_xlabel(sc)
        ax2.set_ylabel('Count')
        
        ax2.set_ylabel('Count')
    plt.tight_layout()  # Adjust layout for better visualization
    png_filename = os.path.join('./Tir', f'cluster2_{cl}.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory
    

InData_Tir_Idx_Plot.to_csv('./Tir/result_Tir.csv')