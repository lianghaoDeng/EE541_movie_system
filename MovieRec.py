import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

movies_path = 'dataset/movies_refined.csv'
ratings_path = 'dataset/ratings_small.csv'

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)


def recommend_similar_movies(cos_similarity, movie_index, k):
    all_index = cos_similarity.iloc[movie_index, :].sort_values(ascending=False).index.to_list()
    rec_index = all_index[1: k+1]
    rec_movie = movies['title'].iloc[rec_index]

    print(f'Watched movie: {movies.title.iloc[movie_index]} \t Genres: {movies.AllGenres.iloc[movie_index]}')
    print(f'Top {k} recommendations:')

    for m, movie in enumerate(rec_movie):
        print(f'{m+1}. {movie} \t Genres: {movies.AllGenres.iloc[rec_index[m]]}')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('distilbert-base-nli-mean-tokens').to(device)

names_and_genres = movies['title'].astype(str) + ',' + movies['AllGenres'].astype(str)
embedding = model.encode(names_and_genres, show_progress_bar=True)

emb_arr = np.array(embedding).astype(np.float16)
cos_similarity = pd.DataFrame(cosine_similarity(emb_arr))

recommend_similar_movies(cos_similarity, movie_index=1, k=5)
