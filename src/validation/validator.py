# src/validation/validator.py

def validate(record):

    required_fields = [
        "timestamp",
        "geo_id",
        "source",
        "entity_type",
        "value"
    ]

    for field in required_fields:

        value = getattr(record, field, None)

        if value is None:
            return False

    try:
        float(record.value)
    except:
        return False

    return True