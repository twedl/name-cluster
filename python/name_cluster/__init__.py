"""name_cluster: cluster business names into entity groups.

See ARCHITECTURE.md for the full design and target public surface.
"""
from __future__ import annotations

from ._lowlevel import normalize  # rust binding
from .cluster import cluster, cluster_names, lsh_calibrate
from .generator import generate_examples
from .score import score_clusters

__all__ = [
    "cluster",
    "cluster_names",
    "lsh_calibrate",
    "normalize",
    "generate_examples",
    "score_clusters",
]

__version__ = "0.1.0"
