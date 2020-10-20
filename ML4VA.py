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
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(0)

df0 = pd.read_csv("Virginia_Crashes.csv")

#%%

# Drop redundant columns
df = df0.drop(["OBJECTID", "Document_Nbr", "Rte_Nm", "Local_Case_Cd", "DIAGRAM", "Node_Info", "X", "Y"], axis = 1)

#%%

# Filter the dataset to only contain crashes from the past two years
indices = np.where(np.logical_or(df["CRASH_YEAR"] == 2019, df["CRASH_YEAR"] == 2020))[0]
df_filt = df.iloc[indices, :].reset_index(drop = True)

#%%

features = {
        "ordinal" : ["Time_Slicing", "Speed_Notspeed", "Belted_Unbelted", "Alcohol_Notalcohol", "Crash_Severity"],
        "nominal" : ["Weather_Condition", "First_Harmful_Event_of_Entire_C", "Collision_Type", "FAC", "FUN", "Light_Condition", "VDOT_District", "Ownership_Used", "Crash_Event_Type_Dsc", "Roadway_Surface_Cond"],
        "numerical": ["Rns_Mp", "K_People", "A_People", "B_People", "C_People", "LATITUDE", "LONGITUDE", "VSP", "SYSTEM", "OWNERSHIP", "Carspeedlimit", "Crash_Military_Tm"]
        }

# Split into X and Y
X = df_filt[features["ordinal"] + features["nominal"] + features["numerical"]]
X.drop(["Crash_Severity"], axis = 1, inplace = True)
Y = df_filt["Crash_Severity"]

X_ordinal = X[features["ordinal"][:-1]]
X_numerical = X[features["numerical"]]

# Remove any rows with missing values for nominal features
X_nominal = X[features["nominal"]]
X_nominal_missing = np.array(X_nominal.isna()).any(axis = 1)
X_nominal_missing_indices = np.where(X_nominal_missing)[0]

X.drop(X_nominal_missing_indices, inplace = True)
X_ordinal.drop(X_nominal_missing_indices, inplace = True)
X_numerical.drop(X_nominal_missing_indices, inplace = True)
X_nominal.drop(X_nominal_missing_indices, inplace = True)
Y.drop(X_nominal_missing_indices, inplace = True)
del X_nominal_missing, X_nominal_missing_indices

#%%
# Ordinal encoding
enc1 = OrdinalEncoder()
X_ordinal_tr = enc1.fit_transform(X_ordinal)

#%%
# One-hot encoding
enc2 = OneHotEncoder()
X_nominal_tr = enc2.fit_transform(X_nominal).toarray()

#%%
# Numerical processing
imp = SimpleImputer(strategy = "median")
X_numerical_imp = imp.fit_transform(X_numerical)
sc = StandardScaler()
X_numerical_tr = sc.fit_transform(X_numerical_imp)

#%%

# Concatenate all of the X arrays
X_tr = np.concatenate((X_numerical_tr, X_ordinal_tr, X_nominal_tr), axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_tr, Y, test_size = 0.2, random_state = 0)

from sklearn.decomposition import PCA
pca = PCA().fit(X_train)
pca_variances = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca_variances)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') # for each component
plt.show()

#%%

pca = PCA(n_components = 63).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#%%
model = SVC()
model.fit(X_train_pca, Y_train)

#%%
Y_train_pred = model.predict(X_train_pca)

#%%
Y_test_pred = model.predict(X_test_pca)

#%%

metrics = {"Training Set": [accuracy_score(Y_train, Y_train), precision_score(Y_train, Y_train), recall_score(Y_train, Y_train), f1_score(Y_train, Y_train)], \
           "Testing Set": [accuracy_score(Y_test, Y_test), precision_score(Y_test, Y_test), recall_score(Y_test, Y_test), f1_score(Y_test, Y_test)]}

metrics = pd.DataFrame(metrics)
metrics.index = ["Accuracy", "Precision", "Recall", "F1 Score"]
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

