import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .vectors import build_wine_vector


def build_wine_matrix(wines, all_flavors, flavor_idf=None):
    wine_vectors = [
        build_wine_vector(wine_row, all_flavors, flavor_idf)
        for _, wine_row in wines.iterrows()
    ]

    if not wine_vectors:
        return np.empty((0, 5 + len(all_flavors)))

    return np.vstack(wine_vectors)


def cosine_similarity_search(query_vector, wine_matrix, top_k=5):
    if wine_matrix.size == 0:
        return []

    similarity_scores = cosine_similarity([query_vector], wine_matrix)[0]
    ranked_indices = np.argsort(similarity_scores)[::-1][:top_k]

    return [
        {
            "row_index": int(row_index),
            "similarity_score": float(similarity_scores[row_index]),
        }
        for row_index in ranked_indices
    ]
