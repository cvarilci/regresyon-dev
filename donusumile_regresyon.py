import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

print(df.describe())



