"""Embedded fake-but-realistic-shaped canonical names for `generate_examples()`.

Used as the default canonical pool when the user passes no `canonicals=`.
Obviously fictitious — no licensing or real-data baggage. Picked to span
common name shapes (single-word brand, multi-word descriptive, possessive,
hyphenated, ampersand, suffix variants, mid-name initialism).

Keep this small (~50 names, ~3 KB inline). Wheel-shipped.
"""

TOY_CANONICALS: list[str] = [
    # Pop-culture fakes (genuinely fictional)
    "Acme Corporation",
    "Initech LLC",
    "Stark Industries",
    "Wayne Enterprises Inc",
    "Pied Piper Holdings",
    "Cyberdyne Systems",
    "Soylent Corporation",
    "Aperture Science Innovators",
    "Tyrell Corp",
    "Globex Corporation",
    "Hooli Inc",
    "Dunder Mifflin Paper Company",
    "Vandelay Industries",
    "Massive Dynamic",
    "Oscorp Industries",
    "Buy n Large",
    # Multi-word descriptive
    "Northern Cascade Manufacturing",
    "Pacific Coast Trading Company",
    "Atlantic Marine Services Inc",
    "Mountainview Logistics LP",
    "Lakeshore Holdings Group",
    "Riverbend Industrial Solutions",
    "Highland Brewing Company",
    "Sunset Valley Vineyards",
    # Hyphen / compound
    "Smith-Henderson Construction",
    "MacArthur-Wallace Holdings",
    "Eastman-Kohl Industries",
    "Northrop-Bayer Manufacturing",
    # Possessive
    "Henderson's Hardware Inc",
    "O'Brien's Trading Co",
    "McAllister's Manufacturing LLC",
    # Ampersand
    "Smith & Wessenburg Industries",
    "Patel & Associates Limited",
    "Walsh & Sons Trading",
    # Initialism / abbreviation forms
    "B&L Manufacturing",
    "JKM Industries Limited",
    "PCM Solutions Inc",
    "RGT Capital Holdings",
    # Geographic prefix (mimics Chinese-style city patterns)
    "Northstar Beijing Trading Co",
    "Pacific Shanghai Logistics LLC",
    # Long brand name (test acronym↔expansion v1.5)
    "International Building Materials Corporation",
    "United Pacific Logistics Group",
    "Continental Shipping & Forwarding Inc",
    "Foothill Industries",
    # Single-word brands (short, hard to fuzzy-match)
    "Brightspoke",
    "Veridica",
    "Lattice",
    "Quantar",
    "Plasmoid",
    "Ferrum",
    "Ironclad",
    "Beacon",
]
