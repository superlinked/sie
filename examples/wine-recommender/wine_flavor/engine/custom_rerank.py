import os

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from .sie_rerank import _resolve_sie_connection, pull_wine_reviews
from .vectors import pull_structure

load_dotenv()


def _structure_label(value):
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium-high"
    if value >= 0.4:
        return "medium"
    if value >= 0.2:
        return "low"
    return "very low"


def _flavor_label(value):
    if value >= 0.8:
        return "very prominent"
    if value >= 0.6:
        return "prominent"
    if value >= 0.4:
        return "medium"
    if value >= 0.2:
        return "subtle"
    return "hint of"


def generate_tasting_note(wine_row):
    from .vectors import pull_flavor_counts

    flavor_counts = pull_flavor_counts(wine_row)
    total_flavor_count = sum(flavor_counts.values())

    flavor_parts = []
    if total_flavor_count > 0:
        for flavor_name, flavor_count in sorted(flavor_counts.items()):
            normalized_count = float(flavor_count) / float(total_flavor_count)
            flavor_parts.append(f"{_flavor_label(normalized_count)} {flavor_name}")

    structure_parts = []
    structure_values = pull_structure(wine_row)
    for structure_name, normalized_value in zip(
        ["acidity", "fizziness", "intensity", "sweetness", "tannin"],
        structure_values,
    ):
        structure_parts.append(f"{_structure_label(normalized_value)} {structure_name}")

    structure_text = ", ".join(structure_parts)
    if flavor_parts:
        flavor_text = ", ".join(flavor_parts)
        return (
            f"This wine had notes of {flavor_text}, "
            f"and structural characteristics including {structure_text}."
        )

    return f"This wine has structural characteristics including {structure_text}."


def generate_user_preferences_note(user_preferences):
    user_flavors = user_preferences["flavors"]
    user_structure = user_preferences["structure"]

    flavor_parts = []
    for flavor_name, flavor_value in sorted(user_flavors.items()):
        flavor_parts.append(f"{_flavor_label(float(flavor_value))} {flavor_name}")

    structure_parts = []
    for structure_name, structure_value in user_structure.items():
        structure_parts.append(f"{_structure_label(float(structure_value))} {structure_name}")

    structure_text = ", ".join(structure_parts)
    if flavor_parts:
        flavor_text = ", ".join(flavor_parts)
        return (
            f"Wine with notes of {flavor_text}, "
            f"and structural characteristics including {structure_text}."
        )

    return f"Wine with structural characteristics including {structure_text}."


class EmbeddingGenerator:
    def __init__(
        self,
        client,
        sample_embedding_dim,
        model_name,
        *,
        gpu=None,
        provision_timeout_s=900,
        a=0.7,
    ):
        self.client = client
        self.model_name = model_name
        self.gpu = gpu
        self.provision_timeout_s = provision_timeout_s
        self.a = a
        self.sample_embedding_dim = sample_embedding_dim

    def _get_embeddings_batch(self, texts):
        if not texts:
            return np.empty((0, self.sample_embedding_dim), dtype=np.float32)

        items = [{"id": f"item-{index}", "text": text} for index, text in enumerate(texts)]
        results = self.client.encode(
            self.model_name,
            items,
            gpu=self.gpu,
            wait_for_capacity=True,
            provision_timeout_s=self.provision_timeout_s,
        )
        embeddings = np.array([result["dense"] for result in results], dtype=np.float32)
        return embeddings

    def _combine_vectors(self, review_vector, tasting_note_vector):
        if review_vector is None and tasting_note_vector is None:
            return np.zeros(self.sample_embedding_dim, dtype=np.float32)
        if review_vector is None:
            return tasting_note_vector
        if tasting_note_vector is None:
            return review_vector
        return (self.a * review_vector) + ((1.0 - self.a) * tasting_note_vector)


def generate_wine_embeddings(embedding_generator, wines_df):
    wines_df = wines_df.copy()

    all_review_texts = []
    review_counts_per_wine = [0] * len(wines_df)
    for local_index, (_, wine_row) in enumerate(wines_df.iterrows()):
        reviews = wine_row["wine_reviews_normalized"]
        review_counts_per_wine[local_index] = len(reviews)
        all_review_texts.extend(reviews)

    batch_review_embeddings = embedding_generator._get_embeddings_batch(all_review_texts)
    batch_tasting_note_embeddings = embedding_generator._get_embeddings_batch(wines_df["tasting_notes"].tolist())

    wines_df["review_embedding"] = None
    wines_df["tasting_note_embedding"] = None

    current_review_embedding_index = 0
    for local_index in range(len(wines_df)):
        wine_index = wines_df.index[local_index]
        review_count = review_counts_per_wine[local_index]
        if review_count > 0:
            wine_review_embeddings = batch_review_embeddings[
                current_review_embedding_index: current_review_embedding_index + review_count
            ]
            wines_df.at[wine_index, "review_embedding"] = np.mean(wine_review_embeddings, axis=0)
            current_review_embedding_index += review_count
        else:
            wines_df.at[wine_index, "review_embedding"] = np.zeros(
                embedding_generator.sample_embedding_dim, dtype=np.float32
            )

        wines_df.at[wine_index, "tasting_note_embedding"] = batch_tasting_note_embeddings[local_index]

    return wines_df


def combine_wine_embeddings(embedding_generator, wines_df):
    wines_df = wines_df.copy()
    wines_df["combined_embedding"] = None

    for local_index, (_, wine_row) in enumerate(wines_df.iterrows()):
        combined_embedding = embedding_generator._combine_vectors(
            wine_row["review_embedding"],
            wine_row["tasting_note_embedding"],
        )
        wines_df.at[wines_df.index[local_index], "combined_embedding"] = combined_embedding

    return wines_df


def generate_user_query_embedding(embedding_generator, user_query_text):
    return embedding_generator._get_embeddings_batch([user_query_text])[0]


def build_final_query_embedding(user_query_embedding, reference_embeddings, alpha):
    if reference_embeddings:
        average_reference_embedding = np.mean(np.vstack(reference_embeddings), axis=0)
    else:
        average_reference_embedding = np.zeros_like(user_query_embedding)

    return (
        alpha * user_query_embedding
        + (1.0 - alpha) * average_reference_embedding
    )


def rerank_wines_with_custom_embeddings(
    wines,
    candidate_row_indices,
    user_preferences,
    *,
    reference_row_indices=None,
    base_url=None,
    model_name=None,
    gpu=None,
    provision_timeout_s=900,
    a=None,
    alpha=None,
    no_review_penalty=0.5,
):
    try:
        from sie_sdk import SIEClient
    except ImportError as exc:
        raise ImportError("sie_sdk is required for custom SIE reranking.") from exc

    base_url, api_key = _resolve_sie_connection(base_url)
    model_name = model_name or os.getenv("SIE_EMBEDDING_MODEL", "BAAI/bge-m3")

    client = SIEClient(base_url, api_key=api_key)
    sample_embedding = client.encode(
        model_name,
        [{"id": "sample", "text": "sample"}],
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )[0]["dense"]
    embedding_generator = EmbeddingGenerator(
        client,
        len(sample_embedding),
        model_name,
        gpu=gpu,
        provision_timeout_s=provision_timeout_s,
        a=a,
    )

    candidate_wines = wines.iloc[candidate_row_indices].copy()
    candidate_wines["tasting_notes"] = candidate_wines.apply(generate_tasting_note, axis=1)
    candidate_wines["wine_reviews_normalized"] = candidate_wines.apply(pull_wine_reviews, axis=1)
    candidate_wines = generate_wine_embeddings(embedding_generator, candidate_wines)
    candidate_wines = combine_wine_embeddings(embedding_generator, candidate_wines)

    reference_embeddings = []
    if reference_row_indices:
        reference_wines = wines.iloc[reference_row_indices].copy()
        reference_wines["tasting_notes"] = reference_wines.apply(generate_tasting_note, axis=1)
        reference_wines["wine_reviews_normalized"] = reference_wines.apply(pull_wine_reviews, axis=1)
        reference_wines = generate_wine_embeddings(embedding_generator, reference_wines)
        reference_wines = combine_wine_embeddings(embedding_generator, reference_wines)
        reference_embeddings = reference_wines["combined_embedding"].tolist()

    user_query_text = generate_user_preferences_note(user_preferences)
    user_query_embedding = generate_user_query_embedding(embedding_generator, user_query_text)
    final_query_embedding = build_final_query_embedding(
        user_query_embedding,
        reference_embeddings,
        alpha,
    )

    wine_scores = []
    for local_index, row_index in enumerate(candidate_row_indices):
        combined_wine_vector = candidate_wines.iloc[local_index]["combined_embedding"]

        score = float(cosine_similarity([final_query_embedding], [combined_wine_vector])[0][0])
        review_count = len(candidate_wines.iloc[local_index]["wine_reviews_normalized"])
        if np.all(candidate_wines.iloc[local_index]["review_embedding"] == 0):
            score *= float(no_review_penalty)

        wine_scores.append(
            {
                "row_index": int(row_index),
                "rerank_score": score,
                "review_count": review_count,
            }
        )

    ranked_wines = sorted(wine_scores, key=lambda item: item["rerank_score"], reverse=True)
    for rank, wine_score in enumerate(ranked_wines):
        wine_score["rerank_rank"] = rank
        wine_score["query_vector_length"] = int(len(final_query_embedding))

    return ranked_wines
