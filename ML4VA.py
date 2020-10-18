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

df = pd.read_csv("Virginia_Crashes.csv")

print(df.head())
print(df['Rd_Type'].value_counts())
print(df.shape)
print(df["Crash_Dt"].value_counts())
print("")
print(df["Time_Slicing"].value_counts())
print("")
print(df["Weather_Condition"].value_counts())
print("")
print(df["First_Harmful_Event_of_Entire_C"].value_counts())
print("")
print(df["Speed_Notspeed"].value_counts())
print("")
print(df["Belted_Unbelted"].value_counts())
print("")
print(df["Alcohol_Notalcohol"].value_counts())
print("")
print(df["Rd_Type"].value_counts())
print("")
print(df["Collision_Type"].value_counts())
print("")
print(df["Vehicle_Body_Type_Cd"].value_counts())
print("")
print(df["Driver_Action_Type_Cd"].value_counts())
print("")
print(df["Crash_Severity"].value_counts())
print("")
print(df["CRASH_YEAR"].value_counts())
print("")
print(df["FAC"].value_counts())
print("")
print(df["FUN"].value_counts())
print("")
print(df["Light_Condition"].value_counts())
print("")
print(df["VDOT_District"].value_counts())
print("")
print(df["Ownership_Used"].value_counts())
print("")
print(df["Physical_Juris"].value_counts())
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

