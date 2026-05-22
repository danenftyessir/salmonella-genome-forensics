"""Metadata cleaning: organism filter, source normalization, class balance checks."""

import pandas as pd

# Controlled vocabulary for isolation_source.
# Key = substring to match (lowercased), Value = normalized label.
# More-specific entries should appear BEFORE shorter ones to avoid
# partial-match shadowing (e.g. "dairy cattle" before "cattle").
SOURCE_NORMALIZATION: dict[str, str] = {
    # poultry
    "broiler carcass": "poultry",
    "broiler cecum": "poultry",
    "broiler feces": "poultry",
    "broiler": "poultry",
    "chicken meat": "poultry",
    "chicken": "poultry",
    "poultry litter": "poultry",
    "poultry meat": "poultry",
    "poultry": "poultry",
    # cattle
    "dairy cattle": "cattle",
    "bovine": "cattle",
    "cattle": "cattle",
    "dairy": "cattle",
    "beef": "cattle",
    "cow": "cattle",
    # swine
    "swine": "swine",
    "pork": "swine",
    "pig": "swine",
    # wild / non-companion animals — MUST appear before generic "feces"
    "rodent": "wildlife",
    "rat ": "wildlife",
    "mouse": "wildlife",
    "llama": "animal_other",
    "camel": "animal_other",
    # human / clinical
    "human clinical": "human",
    "human feces": "human",
    "clinical": "human",
    "patient": "human",
    "human": "human",
    "blood": "human",
    "stool": "human",
    "feces": "human",
    # environment (water)
    "river water": "environment",
    "pond water": "environment",
    "creek": "environment",
    "stream": "environment",
    "river": "environment",
    "pond": "environment",
    "lake": "environment",
    "water": "environment",
    # environment (soil / surface)
    "environmental": "environment",
    "sediment": "environment",
    "surface": "environment",
    "drain": "environment",
    "floor": "environment",
    "soil": "environment",
    "dirt": "environment",
    # produce
    "sprouts": "produce",
    "lettuce": "produce",
    "spinach": "produce",
    "tomato": "produce",
    "cucumber": "produce",
    "pepper": "produce",
    "herbs": "produce",
    "produce": "produce",
    # egg
    "shell egg": "egg",
    "egg": "egg",
    # turkey
    "turkey": "turkey",
    # produce (bean/sprout varieties)
    "sprout": "produce",
    "bean sprout": "produce",
    "mung bean": "produce",
    "pet food": "pet_food",
    # wildlife / wild birds
    "house sparrow": "wildlife",
    "redpoll": "wildlife",
    "sparrow": "wildlife",
    "finch": "wildlife",
    "wild bird": "wildlife",
    "bird": "wildlife",
    # companion animals
    "cat": "companion_animal",
    "feline": "companion_animal",
    "dog": "companion_animal",
    "canine": "companion_animal",
    "kennel": "companion_animal",
    # environment (housing / facility)
    "rabbit cage": "environment",
    "cage": "environment",
    "litter": "environment",
    # human (intestinal / clinical)
    "colon": "human",
    "intestine": "human",
    "cecum": "human",
    "cecal": "human",
    "rectal": "human",
}

AMBIGUOUS_LABELS: set[str] = {
    "unknown", "not provided", "not collected", "missing", "not available",
    "not determined", "na", "n/a", "none", "", "nan",
}

# Binary forensic target: food-chain-associated vs non-food-chain-associated.
# Keyed on NORMALIZED isolation_source values (output of normalize_isolation_source).
# Rationale: Salmonella food safety forensics — distinguish strains that move
# through the food production/distribution chain from those that do not.
BINARY_SOURCE_MAP: dict[str, str] = {
    # ── food_chain_associated ─────────────────────────────────────────────────
    "cattle":           "food_chain_associated",   # beef, dairy, bovine
    "poultry":          "food_chain_associated",   # chicken, broiler
    "swine":            "food_chain_associated",   # pork
    "turkey":           "food_chain_associated",
    "egg":              "food_chain_associated",
    "raw milk":         "food_chain_associated",
    "produce":          "food_chain_associated",   # sprouts, vegetables
    "pet_food":         "food_chain_associated",   # contaminated manufactured food
    "pet food":         "food_chain_associated",
    # ── non_food_chain_associated ─────────────────────────────────────────────
    "human":            "non_food_chain_associated",
    "wildlife":         "non_food_chain_associated",
    "companion_animal": "non_food_chain_associated",
    "environment":      "non_food_chain_associated",
    "animal_other":     "non_food_chain_associated",
}

SOURCE_GROUP_MAP: dict[str, str] = {
    # food-producing animals
    "cattle":        "food_animal",
    "poultry":       "food_animal",
    "swine":         "food_animal",
    "turkey":        "food_animal",
    "egg":           "food_animal",
    # food / food-related (non-animal)
    "produce":          "food_related",
    "raw milk":         "food_related",
    "pet food":         "food_related",
    "pet_food":         "food_related",
    # human / clinical
    "human":            "human_clinical",
    "spiral colon":     "human_clinical",
    # environmental
    "environment":      "environment",
    "kennel":           "environment",
    "rabbit cage":      "environment",
    # companion animals (pet cats, dogs, kennel dogs)
    "companion_animal": "animal_other",
    # wildlife (wild birds, rodents)
    "wildlife":         "animal_other",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_organism(df: pd.DataFrame, organism: str = "Salmonella enterica") -> pd.DataFrame:
    """Keep only rows whose 'organism' column contains `organism`."""
    if "organism" not in df.columns:
        print("[INFO] Kolom 'organism' tidak ditemukan, filter organisme dilewati.")
        return df
    before = len(df)
    df = df[df["organism"].str.contains(organism, case=False, na=False)].reset_index(drop=True)
    print(f"Filter organisme '{organism}': {before} → {len(df)} isolat")
    return df


def drop_missing_accession(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with null or empty assembly_accession."""
    before = len(df)
    mask = df["assembly_accession"].notna() & (df["assembly_accession"].str.strip() != "")
    df = df[mask].reset_index(drop=True)
    print(f"Hapus baris tanpa assembly_accession: {before} → {len(df)} isolat")
    return df


def normalize_isolation_source(df: pd.DataFrame, col: str = "isolation_source") -> pd.DataFrame:
    """Map raw isolation_source strings to controlled vocabulary via SOURCE_NORMALIZATION."""
    if col not in df.columns:
        return df

    def _normalize(val: object) -> object:
        if pd.isna(val):
            return None
        raw = str(val).lower().strip()
        # Exact match first (fast path)
        if raw in SOURCE_NORMALIZATION:
            return SOURCE_NORMALIZATION[raw]
        # Substring match (ordered from specific to generic)
        for pattern, normalized in SOURCE_NORMALIZATION.items():
            if pattern in raw:
                return normalized
        return val  # keep original if no pattern matched

    df = df.copy()
    original = df[col].copy()
    df[col] = df[col].apply(_normalize)
    changed = (df[col] != original).sum()
    print(f"Normalisasi isolation_source: {changed} nilai diubah")
    print(f"  Distribusi: {df[col].value_counts().to_dict()}")
    return df


def remove_ambiguous_sources(df: pd.DataFrame, col: str = "isolation_source") -> pd.DataFrame:
    """Drop rows where isolation_source is in AMBIGUOUS_LABELS or is null."""
    if col not in df.columns:
        return df
    before = len(df)
    mask = df[col].apply(
        lambda v: (pd.notna(v) and str(v).lower().strip() not in AMBIGUOUS_LABELS)
    )
    df = df[mask].reset_index(drop=True)
    print(f"Hapus sumber ambigu '{col}': {before} → {len(df)} isolat")
    return df


def select_dominant_serovars(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Keep only isolates from the top-N most frequent serovars."""
    if "serovar" not in df.columns:
        return df
    top = df["serovar"].value_counts().head(top_n).index.tolist()
    before = len(df)
    df = df[df["serovar"].isin(top)].reset_index(drop=True)
    print(f"Pilih {top_n} serovar dominan {top}: {before} → {len(df)} isolat")
    return df


def add_source_group(df: pd.DataFrame, col: str = "isolation_source") -> pd.DataFrame:
    """Map normalized isolation_source to coarser source_group (5 groups).

    Groups: food_animal, food_related, human_clinical, environment, animal_other.
    Unlisted normalized labels fall back to 'animal_other'.
    """
    df = df.copy()
    df["source_group"] = df[col].map(SOURCE_GROUP_MAP).fillna("animal_other")
    print(f"source_group distribusi: {df['source_group'].value_counts().to_dict()}")
    return df


def add_binary_source(df: pd.DataFrame, col: str = "isolation_source") -> pd.DataFrame:
    """Map normalized isolation_source to binary forensic label.

    Labels
    ------
    food_chain_associated     — livestock (cattle/poultry/swine/turkey), produce,
                                raw milk, pet food.  These are strains that can
                                enter the human food supply.
    non_food_chain_associated — wildlife, companion animals, human clinical,
                                environmental.  Strains not primarily linked to
                                the food production chain.

    Call AFTER normalize_isolation_source() so `col` contains canonical labels.
    Unrecognized values fall back to 'non_food_chain_associated' (conservative).
    """
    df = df.copy()
    df["source_binary"] = df[col].map(BINARY_SOURCE_MAP).fillna("non_food_chain_associated")
    print(f"source_binary distribusi: {df['source_binary'].value_counts().to_dict()}")
    return df


def check_class_balance(df: pd.DataFrame, col: str = "isolation_source") -> None:
    """Print a bar-chart summary of class distribution for a metadata column."""
    if col not in df.columns:
        return
    counts = df[col].value_counts()
    max_count = counts.max()
    print(f"\n[Class Balance] {col}:")
    for label, cnt in counts.items():
        bar = "█" * max(1, int(cnt / max_count * 20))
        print(f"  {str(label):30s} {cnt:4d}  {bar}")
    minority_classes = counts[counts < 3]
    if len(minority_classes):
        print(f"  [WARN] {len(minority_classes)} kelas dengan < 3 sampel: "
              f"{minority_classes.index.tolist()}")
    print()
