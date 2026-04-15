import json
import sqlite3
from pathlib import Path

from . import datasource

DB_PATH = Path("wine_flavor.db")
STATE_PATH = Path("db_creation_state.json")
TARGET_WINE_COUNT = 100_000
BATCH_PAGES = 10
MAX_STALLED_BATCHES = 5

# Vivino type ids seen in this project:
# 1 = red, 2 = white, 3 = sparkling, 4 = rose
QUERY_SLICES = [
    {"country_code": "fr", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 20},
    {"country_code": "fr", "wine_type_ids": [2], "price_range_min": 0, "price_range_max": 20},
    {"country_code": "fr", "wine_type_ids": [3], "price_range_min": 0, "price_range_max": 30},
    {"country_code": "us", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 20},
    {"country_code": "us", "wine_type_ids": [2], "price_range_min": 0, "price_range_max": 20},
    {"country_code": "us", "wine_type_ids": [3], "price_range_min": 0, "price_range_max": 30},
    {"country_code": "ar", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "au", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "au", "wine_type_ids": [2], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "it", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "it", "wine_type_ids": [2], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "es", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "pt", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "za", "wine_type_ids": [1], "price_range_min": 0, "price_range_max": 25},
    {"country_code": "za", "wine_type_ids": [2], "price_range_min": 0, "price_range_max": 25},
    {"country_code": None, "wine_type_ids": [4], "price_range_min": 0, "price_range_max": 25},
]


def _slice_key(slice_config):
    country = slice_config.get("country_code") or "all"
    wine_type = "-".join(str(value) for value in slice_config.get("wine_type_ids") or ["all"])
    price_min = slice_config.get("price_range_min", 0)
    price_max = slice_config.get("price_range_max", 1000)
    return f"{country}|{wine_type}|{price_min}|{price_max}"


def _default_slice_state():
    return {
        "next_page": 1,
        "stalled_batches": 0,
        "status": "not_started",
        "last_error": None,
        "last_unique_count": 0,
    }


def load_state():
    if not STATE_PATH.exists():
        return {
            "current_slice_index": 0,
            "status": "not_started",
            "slice_states": {
                _slice_key(slice_config): _default_slice_state()
                for slice_config in QUERY_SLICES
            },
        }

    state = json.loads(STATE_PATH.read_text())
    state.setdefault("current_slice_index", 0)
    state.setdefault("status", "not_started")
    state.setdefault("slice_states", {})

    for slice_config in QUERY_SLICES:
        state["slice_states"].setdefault(_slice_key(slice_config), _default_slice_state())

    return state


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def connect_db():
    return sqlite3.connect(DB_PATH)


def create_tables(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS wines (
            wine_id INTEGER PRIMARY KEY,
            winery_name TEXT,
            wine_name TEXT,
            vintage_year TEXT,
            rating_average REAL,
            ratings_count INTEGER,
            country_name TEXT,
            region_name TEXT,
            is_natural INTEGER,
            wine_type_id INTEGER,
            taste_acidity REAL,
            taste_fizziness REAL,
            taste_intensity REAL,
            taste_sweetness REAL,
            taste_tannin REAL,
            wine_flavors_json TEXT,
            style_id INTEGER,
            style_name TEXT,
            style_varietal_name TEXT,
            style_body_description TEXT,
            style_acidity_description TEXT,
            style_description TEXT,
            style_food_pairings_json TEXT,
            style_grapes_composition_json TEXT,
            price_amount REAL,
            price_currency TEXT
        )
        """
    )
    connection.commit()


def count_wines(connection):
    row = connection.execute("SELECT COUNT(*) FROM wines").fetchone()
    return int(row[0]) if row else 0


def upsert_wine_batch(connection, wines):
    records = []
    for _, wine_row in wines.iterrows():
        wine_id = wine_row.get("wine_id")
        if wine_id is None:
            continue

        records.append(
            (
                int(wine_id),
                wine_row.get("winery_name"),
                wine_row.get("wine_name"),
                str(wine_row.get("vintage_year")) if wine_row.get("vintage_year") is not None else None,
                wine_row.get("rating_average"),
                wine_row.get("ratings_count"),
                wine_row.get("country_name"),
                wine_row.get("region_name"),
                int(bool(wine_row.get("is_natural"))) if wine_row.get("is_natural") is not None else None,
                wine_row.get("wine_type_id"),
                wine_row.get("taste_acidity"),
                wine_row.get("taste_fizziness"),
                wine_row.get("taste_intensity"),
                wine_row.get("taste_sweetness"),
                wine_row.get("taste_tannin"),
                json.dumps(wine_row.get("wine_flavors") or []),
                wine_row.get("style_id"),
                wine_row.get("style_name"),
                wine_row.get("style_varietal_name"),
                wine_row.get("style_body_description"),
                wine_row.get("style_acidity_description"),
                wine_row.get("style_description"),
                json.dumps(wine_row.get("style_food_pairings") or []),
                json.dumps(wine_row.get("style_grapes_composition") or []),
                wine_row.get("price_amount"),
                wine_row.get("price_currency"),
            )
        )

    connection.executemany(
        """
        INSERT INTO wines (
            wine_id,
            winery_name,
            wine_name,
            vintage_year,
            rating_average,
            ratings_count,
            country_name,
            region_name,
            is_natural,
            wine_type_id,
            taste_acidity,
            taste_fizziness,
            taste_intensity,
            taste_sweetness,
            taste_tannin,
            wine_flavors_json,
            style_id,
            style_name,
            style_varietal_name,
            style_body_description,
            style_acidity_description,
            style_description,
            style_food_pairings_json,
            style_grapes_composition_json,
            price_amount,
            price_currency
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(wine_id) DO UPDATE SET
            winery_name=excluded.winery_name,
            wine_name=excluded.wine_name,
            vintage_year=excluded.vintage_year,
            rating_average=excluded.rating_average,
            ratings_count=excluded.ratings_count,
            country_name=excluded.country_name,
            region_name=excluded.region_name,
            is_natural=excluded.is_natural,
            wine_type_id=excluded.wine_type_id,
            taste_acidity=excluded.taste_acidity,
            taste_fizziness=excluded.taste_fizziness,
            taste_intensity=excluded.taste_intensity,
            taste_sweetness=excluded.taste_sweetness,
            taste_tannin=excluded.taste_tannin,
            wine_flavors_json=excluded.wine_flavors_json,
            style_id=excluded.style_id,
            style_name=excluded.style_name,
            style_varietal_name=excluded.style_varietal_name,
            style_body_description=excluded.style_body_description,
            style_acidity_description=excluded.style_acidity_description,
            style_description=excluded.style_description,
            style_food_pairings_json=excluded.style_food_pairings_json,
            style_grapes_composition_json=excluded.style_grapes_composition_json,
            price_amount=excluded.price_amount,
            price_currency=excluded.price_currency
        """,
        records,
    )
    connection.commit()


def fetch_batch_for_slice(slice_config, page, num_pages):
    return datasource.fetch_vivino_wines(
        page=page,
        num_pages=num_pages,
        price_range_min=slice_config.get("price_range_min", 0),
        price_range_max=slice_config.get("price_range_max", 1000),
        country_code=slice_config.get("country_code"),
        wine_type_ids=slice_config.get("wine_type_ids"),
    )


def main():
    connection = connect_db()
    try:
        print(f"Creating SQLite database at {DB_PATH.resolve()}...", flush=True)
        create_tables(connection)

        state = load_state()
        global_unique_count = count_wines(connection)
        current_slice_index = int(state["current_slice_index"])

        print(
            f"Resuming crawl at slice {current_slice_index + 1}/{len(QUERY_SLICES)}. "
            f"Current unique wines in DB: {global_unique_count}.",
            flush=True,
        )

        while global_unique_count < TARGET_WINE_COUNT and current_slice_index < len(QUERY_SLICES):
            slice_config = QUERY_SLICES[current_slice_index]
            slice_key = _slice_key(slice_config)
            slice_state = state["slice_states"][slice_key]

            current_page = int(slice_state["next_page"])
            stalled_batches = int(slice_state["stalled_batches"])

            print(
                f"Working slice {current_slice_index + 1}/{len(QUERY_SLICES)}: "
                f"{slice_key} starting at page {current_page}.",
                flush=True,
            )

            slice_completed = False
            while global_unique_count < TARGET_WINE_COUNT:
                page_end = current_page + BATCH_PAGES - 1
                print(f"Fetching wine pages {current_page}-{page_end} for {slice_key}...", flush=True)

                try:
                    batch = fetch_batch_for_slice(slice_config, current_page, BATCH_PAGES)
                except RuntimeError as exc:
                    slice_state.update(
                        {
                            "next_page": current_page,
                            "stalled_batches": stalled_batches,
                            "last_unique_count": global_unique_count,
                            "status": "paused",
                            "last_error": str(exc),
                        }
                    )
                    state["current_slice_index"] = current_slice_index
                    state["status"] = "paused"
                    save_state(state)
                    print(f"Paused crawl after request failure: {exc}", flush=True)
                    return

                if batch.empty:
                    slice_completed = True
                    print(f"Slice {slice_key} returned no more wines.", flush=True)
                    break

                upsert_wine_batch(connection, batch)
                new_unique_count = count_wines(connection)
                print(f"Collected {new_unique_count} unique wines so far (target: {TARGET_WINE_COUNT}).", flush=True)

                if new_unique_count == global_unique_count:
                    stalled_batches += 1
                    print(
                        f"No new unique wines found in this batch "
                        f"({stalled_batches}/{MAX_STALLED_BATCHES} stalled batches).",
                        flush=True,
                    )
                    if stalled_batches >= MAX_STALLED_BATCHES:
                        slice_completed = True
                        print(f"Stopping slice {slice_key} because growth has stalled.", flush=True)
                        break
                else:
                    stalled_batches = 0

                global_unique_count = new_unique_count
                current_page += BATCH_PAGES

                slice_state.update(
                    {
                        "next_page": current_page,
                        "stalled_batches": stalled_batches,
                        "last_unique_count": global_unique_count,
                        "status": "in_progress",
                        "last_error": None,
                    }
                )
                state["current_slice_index"] = current_slice_index
                state["status"] = "in_progress"
                save_state(state)

            slice_state.update(
                {
                    "next_page": current_page,
                    "stalled_batches": 0,
                    "last_unique_count": global_unique_count,
                    "status": "completed" if slice_completed else slice_state["status"],
                    "last_error": None if slice_completed else slice_state["last_error"],
                }
            )
            current_slice_index += 1
            state["current_slice_index"] = current_slice_index
            state["status"] = "in_progress"
            save_state(state)

        state["status"] = "completed"
        save_state(state)

        if global_unique_count >= TARGET_WINE_COUNT:
            print(f"Reached target of {TARGET_WINE_COUNT} unique wines.", flush=True)
        else:
            print(
                f"Completed all configured crawl slices with {global_unique_count} unique wines collected.",
                flush=True,
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
