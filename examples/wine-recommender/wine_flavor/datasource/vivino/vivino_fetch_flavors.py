import pandas as pd

from ..http import get_json


# grabs a list of wines
def fetch_vivino_wines(price_range_min=0, price_range_max=1000, page=1, num_pages=1, country_code=None, wine_type_ids=None, min_rating=None):
    url = "https://www.vivino.com/api/explore/explore"
    all_wines = []

    for current_page in range(page, page + num_pages):
        params = {
            "order_by": "price",
            "order": "asc",
            "page": current_page,
            "price_range_max": price_range_max,
            "price_range_min": price_range_min,
        }

        if country_code:
            params["country_code"] = country_code
        if wine_type_ids:
            params["wine_type_ids[]"] = wine_type_ids
        if min_rating:
            params["min_rating"] = min_rating

        payload = get_json(url, params=params)
        matches = payload.get("explore_vintage", {}).get("matches", [])

        if not matches:
            break

        for item in matches:
            vintage = item.get("vintage", {})
            wine = vintage.get("wine", {})
            winery = wine.get("winery", {})
            vintage_stats = vintage.get("statistics", {})
            region = wine.get("region") or {}
            country = region.get("country") or {}
            taste = wine.get("taste") or {}
            structure = taste.get("structure") or {}
            style = wine.get("style") or {}
            price = item.get("price", {})
            currency = price.get("currency") or {}

            food_pairings = [food.get("name") for food in (style.get("foods") or [])]
            grapes_composition = [grape.get("name") for grape in (style.get("grapes") or [])]

            wine_flavors = []
            for flavor_group in (taste.get("flavor") or []):
                flavor_stats = flavor_group.get("stats", {})
                wine_flavors.append({
                    "group": flavor_group.get("group"),
                    "count": flavor_stats.get("count"),
                    "primary_keywords": [
                        {"name": keyword.get("name"), "count": keyword.get("count")}
                        for keyword in (flavor_group.get("primary_keywords") or [])
                    ],
                    "secondary_keywords": [
                        {"name": keyword.get("name"), "count": keyword.get("count")}
                        for keyword in (flavor_group.get("secondary_keywords") or [])
                    ]
                })

            all_wines.append({
                "wine_id": wine.get("id"),
                "winery_name": winery.get("seo_name"),
                "wine_name": wine.get("seo_name"),
                "vintage_year": vintage.get("year"),
                "rating_average": vintage_stats.get("ratings_average"),
                "ratings_count": vintage_stats.get("ratings_count"),
                "country_name": country.get("name"),
                "region_name": region.get("name"),
                "is_natural": wine.get("is_natural"),
                "wine_type_id": wine.get("type_id"),
                "taste_acidity": structure.get("acidity"),
                "taste_fizziness": structure.get("fizziness"),
                "taste_intensity": structure.get("intensity"),
                "taste_sweetness": structure.get("sweetness"),
                "taste_tannin": structure.get("tannin"),
                "wine_flavors": wine_flavors,
                "style_id": style.get("id"),
                "style_name": style.get("name"),
                "style_varietal_name": style.get("varietal_name"),
                "style_body_description": style.get("body"),
                "style_acidity_description": style.get("acidity"),
                "style_description": style.get("description"),
                "style_food_pairings": food_pairings,
                "style_grapes_composition": grapes_composition,
                "price_amount": price.get("amount"),
                "price_currency": currency.get("code")
            })

    return pd.DataFrame(all_wines)
