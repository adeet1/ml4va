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

#%%

df0 = pd.read_csv("Virginia_Crashes.csv")

#%%

# Drop redundant columns
df = df0.drop(["OBJECTID", "Document_Nbr", "Rte_Nm", "Local_Case_Cd", "DIAGRAM", "Node_Info", "X", "Y"], axis = 1)

#%%

features = {
        "ordinal" : ["Time_Slicing", "Speed_Notspeed", "Belted_Unbelted", "Alcohol_Notalcohol", "Crash_Severity"],
        "nominal" : ["Weather_Condition", "First_Harmful_Event_of_Entire_C", "Collision_Type", "FAC", "FUN", "Light_Condition", "VDOT_District", "Ownership_Used", "Crash_Event_Type_Dsc", "Roadway_Surface_Cond"],
        "numerical": ["Rns_Mp", "K_People", "A_People", "B_People", "C_People", "LATITUDE", "LONGITUDE", "VSP", "SYSTEM", "OWNERSHIP", "Carspeedlimit", "Crash_Military_Tm", "Driverage"]
        }

pipeline_num = Pipeline([("imputer", SimpleImputer(strategy = "median")), ("scaler", StandardScaler())])

pipeline = ColumnTransformer([("numerical", pipeline_num, features["numerical"]), ("ordinal", OrdinalEncoder(), features["ordinal"]), ("nominal", OneHotEncoder(), features["nominal"])])

df_tr = pipeline.fit_transform(df)

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

