import pandas as pd


def build_results_frame(wines, reranked_matches):
    match_frame = pd.DataFrame(reranked_matches)
    result_frame = match_frame.merge(
        wines.reset_index().rename(columns={"index": "row_index"})[
            [
                "row_index",
                "wine_id",
                "wine_name",
                "winery_name",
                "vintage_year",
                "rating_average",
                "country_name",
                "region_name",
                "price_amount",
                "price_currency",
                "review_count",
            ]
        ],
        on="row_index",
        how="left",
        suffixes=("", "_wine"),
    ).sort_values(["rerank_rank", "rerank_score"], ascending=[True, False])

    if "review_count_wine" in result_frame.columns:
        result_frame = result_frame.drop(columns=["review_count_wine"])

    return result_frame


def print_top_results(top_results):
    print("\n--- Top Wine Recommendations (review rerank) ---")
    display_frame = top_results.copy()
    if "rerank_score" in display_frame.columns:
        display_frame["rerank_score"] = display_frame["rerank_score"].round(4)
    print(display_frame.to_string(index=False))
