import chromadb
import random

from typing import Dict, Any, Union

db_client = chromadb.PersistentClient(path="./blue_nile_agentic_db")
product_collection = db_client.get_collection(name="product_knowledge")
visual_collection = db_client.get_collection(name="visual_index")


def check_product_availability(filters):
    """
    Check if products exist in ChromaDB matching a set of metadata filters.
    """
    active_filters = []

    for k, v in filters.items():
        if v is None or v == "" or (isinstance(v, str) and v.lower == "none"):
            continue

        if k == "price" and isinstance(v, dict):
            if "min" in v and v["min"] is not None:
                active_filters.append({"price": {"$gte": float(v["min"])}})
            if "max" in v and v["max"] is not None:
                active_filters.append({"price": {"$lte": float(v["max"])}})

        elif isinstance(v, list):
            active_filters.append({k: {"$in": v}})
        else:
            active_filters.append({k: v})

    if len(active_filters) > 1:
        where_clause = {"$and": active_filters}
    else:
        where_clause = active_filters[0]

    results = product_collection.get(
        where=where_clause,
        limit=1
    )

    count = len(results["ids"])
    exists = count > 0

    return {
        "exists": exists,
        "count": count,
        "reason": f"Found matching items for {active_filters}" if exists else f"No items found for {active_filters}"
    }


def get_unique_values(field_name):
    results = product_collection.get(include=["metadatas"])
    values = set()
    for m in results["metadatas"]:
        if m and field_name in m:
            v = m[field_name]
            if v and v != "Unknown":
                values.add(v)
    return list(values)


def get_smart_gallery(attribute_name, available_options, current_filters, limit=6):
    gallery_items = []

    def build_query(base_query):
        if not current_filters:
            return base_query

        combined = {"$and": [base_query]}

        for k, v in current_filters.items():
            if not v or k == attribute_name:
                continue

            if isinstance(v, list):
                if len(v) == 1:
                    combined["$and"].append({k: v[0]})
                else:
                    # ChromaDB syntax for "Key is IN [A, B]"
                    combined["$and"].append({k: {"$in": v}})
            else:
                combined["$and"].append({k: v})

        return combined

    if attribute_name == "price":
        price_buckets = [
            ("Under $1k", 0, 1000),
            ("$1k - $2k", 1000, 2000),
            ("$2k - $4k", 2000, 4000),
            ("$4k - $8k", 4000, 8000),
            ("$8k+", 8000, 500000)
        ]

        for label, min_p, max_p in price_buckets:
            price_query = {"$and": [
                {"price": {"$gte": 0}},
                {"price": {"$lte": 1000}}
            ]}

            final_query = build_query(price_query)
            item = _fetch_random_product(final_query, label, attribute_name)
            if item:
                gallery_items.append(item)
    else:
        shuffled_options = random.sample(
            available_options, len(available_options))

        for option in shuffled_options:
            attr_query = {attribute_name: option}

            final_query = build_query(attr_query)

            item = _fetch_random_product(
                final_query, option, attribute_name)
            if item:
                gallery_items.append(item)

        if len(gallery_items) > limit:
            gallery_items = gallery_items[:limit]
    return gallery_items


def _fetch_random_product(query, label_value, attribute_name):
    try:
        product_results = product_collection.get(
            where=query,
            limit=5,
            include=["metadatas"]
        )
    except Exception as e:
        print(f"Gallery Query Error: {e}")
        return None

    if not product_results["ids"]:
        return None

    random_idx = random.randint(0, len(product_results["ids"]) - 1)
    pid = product_results["metadatas"][random_idx]["product_id"]
    p_name = product_results["metadatas"][random_idx]["name"]
    p_price = product_results["metadatas"][random_idx]["price"]

    image_results = visual_collection.get(
        where={"$and": [{"parent_id": pid}, {"view_index": 0}]},
        limit=1,
        include=["metadatas"]
    )

    if not image_results["ids"]:
        image_results = visual_collection.get(
            where={"parent_id": pid}, limit=1, include=["metadatas"])

    if image_results["metadatas"]:
        return {
            "attribute": attribute_name,
            "value": label_value,
            "actual_price": p_price,
            "name": p_name,
            "image_url": image_results["metadatas"][0]["image_url"]
        }
    return None
