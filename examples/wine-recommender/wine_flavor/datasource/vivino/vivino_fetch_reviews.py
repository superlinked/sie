from ..http import get_json


def fetch_vivino_reviews(wine_id, year=None, page=1, num_pages=1, per_page=50, language=None):
    all_reviews = []

    for current_page in range(page, page + num_pages):
        params = {
            "page": current_page,
            "per_page": per_page,
        }
        if year is not None:
            params["year"] = year

        payload = get_json(
            f"https://www.vivino.com/api/wines/{wine_id}/reviews",
            params=params,
        )
        reviews = payload.get("reviews", [])
        if not reviews:
            break

        for review in reviews:
            review_record = {
                "review_id": review.get("id"),
                "wine_id": wine_id,
                "language": review.get("language"),
                "rating": review.get("rating"),
                "note": review.get("note"),
                "created_at": review.get("created_at"),
                "author": (review.get("user") or {}).get("username"),
            }
            if language is None or review_record["language"] == language:
                all_reviews.append(review_record)

    return all_reviews


def attach_vivino_reviews(wines, review_pages=1, reviews_per_page=50, language=None):
    wines = wines.copy()
    review_payloads = []

    for _, wine_row in wines.iterrows():
        wine_id = wine_row.get("wine_id")
        vintage_year = wine_row.get("vintage_year")
        if wine_id is None:
            review_payloads.append([])
            continue

        wine_reviews = fetch_vivino_reviews(
            int(wine_id),
            year=vintage_year,
            num_pages=review_pages,
            per_page=reviews_per_page,
            language=language,
        )
        review_payloads.append(wine_reviews)

    wines["wine_reviews"] = review_payloads
    wines["review_count"] = [len(reviews) for reviews in review_payloads]
    return wines
