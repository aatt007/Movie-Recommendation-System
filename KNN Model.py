import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
Collaborative Filtering
After preprocessing, the process for creating a User Based recommendation system is as follows:
Select a user with the movies the user has watched
Based on his rating to movies, find the top X neighbours
Get the watched movie record of the user for each neighbour.
Calculate a similarity score using some formula
Recommend the items with the highest score
'''

df_mr=pd.read_csv('movie_mr.csv')
df_mr=df_mr.drop(labels=['genres','timestamp'], axis=1)
#print(df_mr)
#print(df_mr.info())
#print(df_mr.isnull().sum())

df_mr_aver_rating=df_mr.groupby('title')['rating'].mean().sort_values(ascending=False).reset_index().rename(columns={'rating':'Average Rating'})
print(df_mr_aver_rating.head())

df_mr_rating_count=df_mr.groupby('title')['rating'].count().sort_values(ascending=True).reset_index().rename(columns={'rating':'Rating Count'})
print(df_mr_rating_count.head())

df_mr_count_aver=df_mr_aver_rating.merge(df_mr_rating_count, on='title')
print(df_mr_count_aver.head())

'''
Observation-
Many movies have a pefrect 5 star average rating on a dataset of almost 100k user ratings. This suggests the existence of outliers which we need to further confirm with visualization.
The presence of single ratings for several movies suggests that I set a threshold value of ratings to produce valuable recommendations
'''
'''
sns.set(font_scale=1)
plt.rcParams["axes.grid"] = False
plt.style.use('dark_background')

plt.figure(figsize=(12,4))
plt.hist(df_mr_count_aver['Rating Count'],bins=80,color='tab:purple')
plt.ylabel('Ratings Count(Scaled)', fontsize=16)
plt.savefig('ratingcounthist.jpg')

plt.figure(figsize=(12,4))
plt.hist(df_mr_count_aver['Average Rating'],bins=80,color='tab:purple')
plt.ylabel('Average Rating',fontsize=16)
plt.savefig('avgratinghist.jpg')
plt.show()

plot=sns.jointplot(x='Average Rating',y='Rating Count',data=df_mr_count_aver,alpha=0.5, color='tab:pink')
plot.savefig('joinplot.jpg')
plt.show()
'''

df_mr_ratingcount=df_mr.merge(df_mr_rating_count, left_on='title', right_on='title', how='left')
print(df_mr_ratingcount.head())
print(df_mr_ratingcount['Rating Count'].describe())

popularity_threshold = 50
popular_movies= df_mr_ratingcount[df_mr_ratingcount['Rating Count']>=popularity_threshold]
print(popular_movies.head())
print(popular_movies.shape)

import os
movie_features_df=popular_movies.pivot_table(index='title',columns='userId',values='rating').fillna(0)
print(movie_features_df)

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(movie_features_df.values)
print(movie_features_df_matrix)

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)
NearestNeighbors(algorithm='brute', metric='cosine')
print(movie_features_df.shape)

query_index = np.random.choice(movie_features_df.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
query_index=196

print(movie_features_df.head())

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))