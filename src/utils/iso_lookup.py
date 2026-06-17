import country_converter as coco


def get_iso3(country_name):
    try:
        return coco.convert(
            names=country_name,
            to="ISO3"
        )
    except Exception:
        return None


if __name__ == "__main__":
    print(get_iso3("India"))
    print(get_iso3("Ukraine"))
    print(get_iso3("United States"))