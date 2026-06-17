import hashlib


def generate_record_key(
    source,
    geo_id,
    geo_sub,
    entity_type,
    timestamp,
    metadata_signature
):
    raw = (
        f"{source}|{geo_id}|{geo_sub}|"
        f"{entity_type}|{timestamp}|"
        f"{metadata_signature}"
    )

    return hashlib.sha1(
        raw.encode()
    ).hexdigest()


if __name__ == "__main__":

    key = generate_record_key(
        source="acled",
        geo_id="UKR",
        geo_sub="Kyiv",
        entity_type="conflict_event",
        timestamp="2025-01-01",
        metadata_signature="Violence"
    )

    print(key)