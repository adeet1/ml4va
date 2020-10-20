import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import os 
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(0)

#%%
df0 = pd.read_csv("Virginia_Crashes.csv")

#%%

# Drop redundant columns
df = df0.drop(["OBJECTID", "Document_Nbr", "Rte_Nm", "Local_Case_Cd", "DIAGRAM", "Node_Info", "X", "Y"], axis = 1)

# Filter the dataset to only contain crashes from the past two years
indices = np.where(np.logical_or(df["CRASH_YEAR"] == 2019, df["CRASH_YEAR"] == 2020))[0]
df_filt = df.iloc[indices, :].reset_index(drop = True)

variables = {
        "ordinal" : ["Time_Slicing", "Speed_Notspeed", "Belted_Unbelted", "Alcohol_Notalcohol"],
        "nominal" : ["Weather_Condition", "First_Harmful_Event_of_Entire_C", "Collision_Type", "FAC", "FUN", "Light_Condition", "VDOT_District", "Ownership_Used", "Crash_Event_Type_Dsc", "Roadway_Surface_Cond"],
        "numerical": ["Rns_Mp", "K_People", "A_People", "B_People", "C_People", "LATITUDE", "LONGITUDE", "VSP", "SYSTEM", "OWNERSHIP", "Carspeedlimit", "Crash_Military_Tm"],
        "target": ["Crash_Severity"]
        }

# Split into X and Y
X = df_filt[variables["ordinal"] + variables["nominal"] + variables["numerical"]]
Y = df_filt["Crash_Severity"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
Y_train = Y_train.reset_index(drop = True)
Y_test = Y_test.reset_index(drop = True)

X_train_ordinal = X_train[variables["ordinal"]]
X_train_numerical = X_train[variables["numerical"]]
X_test_ordinal = X_test[variables["ordinal"]]
X_test_numerical = X_test[variables["numerical"]]

# Remove any rows with missing values for nominal features
X_train_nominal = X_train[variables["nominal"]]
X_train_nominal_missing = np.array(X_train_nominal.isna()).any(axis = 1)
X_train_nominal_missing_indices = np.where(X_train_nominal_missing)[0]
X_train.drop(X_train_nominal_missing_indices, inplace = True)
X_train_ordinal.drop(X_train_nominal_missing_indices, inplace = True)
X_train_numerical.drop(X_train_nominal_missing_indices, inplace = True)
X_train_nominal.drop(X_train_nominal_missing_indices, inplace = True)
Y_train.drop(X_train_nominal_missing_indices, inplace = True)
del X_train_nominal_missing, X_train_nominal_missing_indices

X_test_nominal = X_test[variables["nominal"]]
X_test_nominal_missing = np.array(X_test_nominal.isna()).any(axis = 1)
X_test_nominal_missing_indices = np.where(X_test_nominal_missing)[0]

X_test.drop(X_test_nominal_missing_indices, inplace = True)
X_test_ordinal.drop(X_test_nominal_missing_indices, inplace = True)
X_test_numerical.drop(X_test_nominal_missing_indices, inplace = True)
X_test_nominal.drop(X_test_nominal_missing_indices, inplace = True)
Y_test.drop(X_test_nominal_missing_indices, inplace = True)
del X_test_nominal_missing, X_test_nominal_missing_indices

# Ordinal encoding 
X_train_ordinal_tr = OrdinalEncoder().fit_transform(X_train_ordinal)
X_test_ordinal_tr = OrdinalEncoder().fit_transform(X_test_ordinal)

# One-hot encoding
enc2 = OneHotEncoder()
X_train_nominal_tr = enc2.fit_transform(X_train_nominal).toarray()
X_test_nominal_tr = enc2.transform(X_test_nominal).toarray()

# Numerical processing
X_train_numerical = SimpleImputer(strategy = "median").fit_transform(X_train_numerical)
X_test_numerical = SimpleImputer(strategy = "median").fit_transform(X_test_numerical)
sc = StandardScaler()
X_train_numerical_tr = sc.fit_transform(X_train_numerical)
X_test_numerical_tr = sc.transform(X_test_numerical)

# Label encoding
Y_train_tr = LabelEncoder().fit_transform(Y_train)
Y_test_tr = LabelEncoder().fit_transform(Y_test)

# Concatenate all of the X arrays
X_train_numerical_tr = pd.DataFrame(X_train_numerical_tr)
X_train_ordinal_tr = pd.DataFrame(X_train_ordinal_tr)
X_train_nominal_tr = pd.DataFrame(X_train_nominal_tr)
X_train_tr = pd.concat((X_train_numerical_tr, X_train_ordinal_tr, X_train_nominal_tr), axis = 1)

X_test_numerical_tr = pd.DataFrame(X_test_numerical_tr)
X_test_ordinal_tr = pd.DataFrame(X_test_ordinal_tr)
X_test_nominal_tr = pd.DataFrame(X_test_nominal_tr)
X_test_tr = pd.concat((X_test_numerical_tr, X_test_ordinal_tr, X_test_nominal_tr), axis = 1)

del X_train_numerical, X_train_ordinal, X_train_nominal
del X_train_numerical_tr, X_train_ordinal_tr, X_train_nominal_tr
del X_test_numerical, X_test_ordinal, X_test_nominal
del X_test_numerical_tr, X_test_ordinal_tr, X_test_nominal_tr
del X_train, X_test, X, Y_train, Y_test, Y

#%%
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train_tr)
pca_variances = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca_variances)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') # for each component
plt.show()

n_pc = np.where(pca_variances >= 0.95)[0][0] + 1

pca = PCA(n_components = n_pc).fit(X_train_tr)
X_train_pca = pca.transform(X_train_tr)
X_test_pca = pca.transform(X_test_tr)

#%%
model = SVC()
model.fit(X_train_pca, Y_train_tr)

#%%
Y_train_pred = model.predict(X_train_pca)
Y_test_pred = model.predict(X_test_pca)

#%%
acc_train = accuracy_score(Y_train_tr, Y_train_pred)
prec_train = precision_score(Y_train_tr, Y_train_pred, average = None)
recall_train = recall_score(Y_train_tr, Y_train_pred, average = None)
f1_train = f1_score(Y_train_tr, Y_train_pred, average = None)

acc_test = accuracy_score(Y_test_tr, Y_test_pred)
prec_test = precision_score(Y_test_tr, Y_test_pred, average = None)
recall_test = recall_score(Y_test_tr, Y_test_pred, average = None)
f1_test = f1_score(Y_test_tr, Y_test_pred, average = None)

metrics_train = np.array([acc_train, prec_train[0], prec_train[1], prec_train[2], prec_train[3], prec_train[4], recall_train[0], recall_train[1], recall_train[2], recall_train[3], recall_train[4], f1_train[0], f1_train[1], f1_train[2], f1_train[3], f1_train[4]])

metrics_test = np.array([acc_test, prec_test[0], prec_test[1], prec_test[2], prec_test[3], prec_test[4], recall_test[0], recall_test[1], recall_test[2], recall_test[3], recall_test[4], f1_test[0], f1_test[1], f1_test[2], f1_test[3], f1_test[4]])

metrics = {"Training Set": metrics_train, "Testing Set": metrics_test}
metrics = pd.DataFrame(metrics)
metrics.index = ["Accuracy", "Precision (Class 0)", "Precision (Class 1)", "Precision (Class 2)", "Precision (Class 3)", "Precision (Class 4)", "Recall (Class 0)", "Recall (Class 1)", "Recall (Class 2)", "Recall (Class 3)", "Recall (Class 4)", "F1 Score (Class 0)", "F1 Score (Class 1)", "F1 Score (Class 2)", "F1 Score (Class 3)", "F1 Score (Class 4)"]
print(metrics)

#%%

# Drop: Crash_Dt, CRASH_YEAR
# We could feature engineer Light_Condition
# We could group regions (Physical_Juris) together into Northern Virginia, Central Virginia, urban, rural, etc.
# Need to see if every city in Virginia has a corresponding VDOT_District
# Need to choose between one of: Physical_Juris, Plan_District, VDOT_District
# Need to choose between one of: FUN, Rte_Category_Cd

#print(df["Crash_Dt"].value_counts())
print("")
#print(df["Time_Slicing"].value_counts())
print("")
#print(df["Weather_Condition"].value_counts())
print("")
#print(df["First_Harmful_Event_of_Entire_C"].value_counts())
print("")
#print(df["Speed_Notspeed"].value_counts())
print("")
#print(df["Belted_Unbelted"].value_counts())
print("")
#print(df["Alcohol_Notalcohol"].value_counts())
print("")
print(df["Rd_Type"].value_counts())
print("")
#print(df["Collision_Type"].value_counts())
#print("")
print(df["Vehicle_Body_Type_Cd"].value_counts())
print("")
print(df["Driver_Action_Type_Cd"].value_counts())
print("")
#print(df["Crash_Severity"].value_counts())
print("")
print(df["CRASH_YEAR"].value_counts())
print("")
#print(df["FAC"].value_counts())
print("")
#print(df["FUN"].value_counts())
print("")
#print(df["Light_Condition"].value_counts())
print("")
#print(df["VDOT_District"].value_counts())
print("")
#print(df["Ownership_Used"].value_counts())
print("")
#print(df["Physical_Juris"].value_counts())
print("")
print(df["Plan_District"].value_counts())
print("")
print(df["Crash_Event_Type_Dsc"].value_counts())
print("")
print(df["Rte_Category_Cd"].value_counts())
print("")
print(df["Roadway_Surface_Cond"].value_counts())
print("")
print(df["Drivergen"].value_counts())
print("")
print(df["Driverinjurytype"].value_counts())
print("")
print(df["Pedgen"].value_counts())
print("")
print(df["Pedinjurytype"].value_counts())
print("")
print(df["Passgen"].value_counts())
print("")
print(df["Passinjurytype"].value_counts())

#%%

