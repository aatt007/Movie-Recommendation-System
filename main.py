import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df_m=pd.read_csv('movies.csv')
print(df_m)
print(df_m.info())

df_r=pd.read_csv('ratings.csv')
print(df_r)
print(df_r.info())

df_m_r=df_m.merge(df_r, on=['movieId'], how='inner')
print(df_m_r)
print(df_m_r.info())
df_m_r.to_csv('movie_mr.csv')
df_mr=pd.read_csv('movie_mr.csv')
print(df_mr)