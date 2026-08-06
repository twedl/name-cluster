"""namecluster: cluster business names into entity groups.

See ARCHITECTURE.md for the full design and target public surface.
"""

from __future__ import annotations

from .acronym import acronym_map
from .cluster import cluster, cluster_names, lsh_calibrate
from .debug import candidates, explain
from .generator import generate_examples
from .normalize import normalize, normalize_names
from .score import score_clusters

__all__ = [
    "acronym_map",
    "candidates",
    "cluster",
    "cluster_names",
    "explain",
    "generate_examples",
    "lsh_calibrate",
    "normalize",
    "normalize_names",
    "score_clusters",
]

__version__ = "0.1.1"
