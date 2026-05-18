"""Sequence-level helpers: base validation, GC content."""

VALID_BASES = set("ATGCN")


def validate_sequence(seq: str, valid_bases: set = None) -> str:
    """Replace non-standard bases with 'N'."""
    if valid_bases is None:
        valid_bases = VALID_BASES
    return "".join(b if b.upper() in valid_bases else "N" for b in seq.upper())


def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    total = len([b for b in seq if b in "ATGC"])
    return gc / total if total > 0 else 0.0


def has_ambiguous(seq: str, threshold: float = 0.05) -> bool:
    """True if fraction of N bases exceeds threshold."""
    seq = seq.upper()
    return seq.count("N") / len(seq) > threshold if seq else True
