# ---------- Data filtering function ----------
def filter_data(df, country=None, start_year=None, end_year=None):
    """Filter the dataset by country and year range."""
    if country:
        df = df[df["country"] == country]
    if start_year and end_year:
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    return df