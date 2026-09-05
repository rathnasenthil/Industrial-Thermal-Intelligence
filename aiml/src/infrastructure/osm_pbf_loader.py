"""
Streaming OSM PBF ingestion for GIFT Stage I.1.

This module extends the static-extract loaders in `osm_loader.py` (which
handle GeoJSON/CSV exports) to also support a real, India-wide OSM PBF
file (e.g. ``aiml/data/raw/india-260904.osm.pbf``), WITHOUT ever loading
the whole country into memory as a single GeoDataFrame.

DEPENDENCY CHOICE: pyosmium (PyPI package name ``osmium``)
------------------------------------------------------------------------
pyosmium is the standard Python binding for libosmium, the C++ library
used by most production OSM tooling (osm2pgsql, osmium-tool, etc.). It
was selected because:

* It provides official pre-built wheels for Windows 64-bit (this
  project's development environment), Linux and macOS -- no compiler
  toolchain required to install it.
* It is a genuine *streaming* SAX-style parser: `osmium.SimpleHandler`
  callbacks (`node`/`way`/`relation`) fire one object at a time as the
  file is scanned, so memory use is controlled entirely by what *this*
  code chooses to retain -- not by the size of the input file.
* It is the smallest reasonable dependency capable of iterating nodes,
  ways AND relations without materializing the whole file; alternatives
  either wrap the same library (e.g. `pyrosm`, which lacks official
  Windows wheels) or require loading everything into memory first
  (`osmnx`/`geopandas` GeoJSON/Shapefile readers, which is exactly what
  this task forbids for an India-wide extract).

STREAMING / MEMORY STRATEGY
------------------------------------------------------------------------
    PBF file
      -> read one node/way/relation at a time (libosmium, C++)
      -> cheap tag-key/value check (`_is_relevant_tags`) BEFORE building
         any Python dict or geometry -- the vast majority of OSM objects
         in an India-wide extract (plain road/building/amenity nodes and
         ways) are rejected here at negligible cost.
      -> only objects that pass the check are converted into a small
         Python record (tags dict + geometry) and appended to a list.
      -> that list (expected to be a tiny fraction of the country-wide
         object count) becomes the GeoDataFrame handed to
         `osm_normalization.normalize_osm_facilities`, exactly like the
         GeoJSON/CSV loaders already do.

Resolving way node coordinates requires a location index (nodes must be
cached as they stream by so that a *later* way referencing them can look
up their position); this uses libosmium's adaptive ``flex_mem`` index
(dense array for the common contiguous id ranges, sparse map for the
rest), which is the default and recommended index for country-sized
extracts. This is the one part of the pipeline whose memory use scales
with the *total* node count of the input file, not just the candidate
count -- an unavoidable cost of resolving way geometry from a PBF
without a second, pre-built spatial database.

CANDIDATE TAG FILTER (deliberately conservative, see `_is_relevant_tags`)
------------------------------------------------------------------------
The filter mirrors the *load-bearing* tag evidence that
`osm_normalization.classify_facility_type` actually inspects
(``industrial=*``, ``landuse=industrial``/``quarry``, ``power=plant``/
``generator``, ``man_made=works``/``wastewater_plant``/``storage_tank``/
``tank``, ``content=lng``/``gas``/``natural_gas``), so nothing that would
be classified as a supported `facility_type` is filtered out here.

One deliberate exception: ``power=*`` is narrowed to
``{"plant", "generator"}`` rather than "any power tag". The existing
(frozen) normalization code treats ANY ``power=*`` value as
OTHER_INDUSTRIAL evidence when nothing more specific matches -- which is
correct and unchanged for GeoJSON/CSV inputs -- but taken literally
against an India-wide PBF this would also pull in every transmission
tower, pole, line and substation in the country (a very large object
count with no facility-scale meaning) purely as *candidates* for this
loader. Restricting the PBF-side candidate filter to generation-type
power tags is a pragmatic, documented scope limitation for THIS loader's
early filtering step (not a change to `classify_facility_type` itself,
which is untouched).

RELATIONS / MULTIPOLYGONS -- KNOWN LIMITATION
------------------------------------------------------------------------
Full multipolygon relation geometry reconstruction (resolving every
outer/inner member way's node ring via libosmium's two-pass
`osmium.area.AreaManager`) is NOT implemented here. That machinery
requires a second full pass over the file assembling areas for every
closed way and multipolygon relation in the country (not just candidate
industrial ones, since area assembly happens before any tag filtering
is possible) -- a large, indiscriminate cost for a benefit limited to
the minority of facilities mapped as multi-ring relations (most
real-world industrial/power/mining facilities are simple closed ways,
which ARE fully supported below).

Instead, a relation with relevant tags is preserved with its
`osm_id`/`osm_type`/`raw_tags`/`name`, but `geometry=None`. It flows
through the same normalization/validation pipeline as everything else,
where the existing (unmodified) `facility_validation.validate_facilities`
correctly flags it as `invalid_geometry` and it is preserved -- with a
`rejection_reason` -- in the rejected-records output, never silently
dropped. This is exactly the fallback behavior requested for relations
that cannot be safely reconstructed.

WAYS
------------------------------------------------------------------------
A candidate way's geometry is built directly from its member nodes'
resolved locations (no `AreaManager` needed):

* If every member node's location resolved, and the way is closed
  (first node id == last node id) with at least 4 points -> `Polygon`.
* If every member node's location resolved but the way is open -> a
  `LineString`. `LineString` is not in
  `facility_schema.SUPPORTED_GEOMETRY_TYPES`, so -- exactly like a
  relation -- this is consistently flagged as `invalid_geometry` and
  preserved in the rejected output rather than silently dropped or
  faked into a Polygon it doesn't represent.
* If any member node's location failed to resolve (e.g. the extract's
  bounding box clipped a node the way references) -> `geometry=None`,
  also preserved/rejected with a reason.

Preserving the ORIGINAL polygon geometry (not its centroid) for
polygon-shaped ways is handled identically to the GeoJSON/CSV loaders:
this module only ever returns the full geometry; the representative
point used for spatial-association purposes is computed later, in
`osm_normalization.normalize_osm_facilities`, exactly as for any other
input format.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

try:
    import osmium
except ImportError as exc:  # pragma: no cover - exercised only if dependency missing
    raise ImportError(
        "OSM PBF ingestion requires the 'osmium' package (pyosmium). "
        "Install it with: pip install osmium"
    ) from exc

# Default location-index type for resolving way node coordinates. "flex_mem"
# is libosmium's adaptive in-memory index (dense array for contiguous id
# ranges, sparse map otherwise) and is the recommended choice for
# country-sized extracts -- see module docstring. A disk-backed type such as
# "dense_file_array,<path>" can be passed explicitly to `load_osm_pbf` for
# more memory-constrained environments, at the cost of slower way resolution.
DEFAULT_LOCATION_INDEX = "flex_mem"

# power=* values treated as candidate evidence by THIS loader's early
# filter. Deliberately narrower than "any power tag" -- see module
# docstring ("RELATIONS / MULTIPOLYGONS" section above covers relations;
# this note is about power=* specifically).
_CANDIDATE_POWER_VALUES = frozenset({"plant", "generator"})

# landuse=* values treated as candidate evidence.
_CANDIDATE_LANDUSE_VALUES = frozenset({"industrial", "quarry"})

# man_made=* values treated as candidate evidence (matches the exact set
# `classify_facility_type` inspects: OTHER_INDUSTRIAL for works/
# wastewater_plant, LNG gas-evidence for storage_tank/tank).
_CANDIDATE_MAN_MADE_VALUES = frozenset({"works", "wastewater_plant", "storage_tank", "tank"})

# content=* values treated as candidate evidence (LNG gas-evidence, see
# `classify_facility_type`).
_CANDIDATE_CONTENT_VALUES = frozenset({"lng", "gas", "natural_gas"})


def _is_relevant_tags(tags: Any) -> bool:
    """Cheap, early check: does this object carry any candidate industrial tag?

    Operates directly on the raw `osmium.osm.TagList` (via `.get`) so that
    the overwhelming majority of plain OSM objects in an India-wide
    extract (untagged way-geometry nodes, roads, amenities, residential
    buildings, etc.) are rejected without ever building a Python dict or
    geometry for them.
    """
    if len(tags) == 0:
        return False
    if tags.get("industrial") is not None:
        return True
    power = tags.get("power")
    if power is not None and power.lower() in _CANDIDATE_POWER_VALUES:
        return True
    landuse = tags.get("landuse")
    if landuse is not None and landuse.lower() in _CANDIDATE_LANDUSE_VALUES:
        return True
    man_made = tags.get("man_made")
    if man_made is not None and man_made.lower() in _CANDIDATE_MAN_MADE_VALUES:
        return True
    content = tags.get("content")
    if content is not None and content.lower() in _CANDIDATE_CONTENT_VALUES:
        return True
    return False


def _peak_working_set_mb() -> float | None:
    """Best-effort peak process memory (MB) since process start, no extra dependency.

    Uses the Windows ``psapi``/``kernel32`` APIs via `ctypes` (this
    project's development/CI environment is Windows) with a `resource`
    (POSIX) fallback for portability. Returns ``None`` if neither is
    available -- deliberately honest about "if measurable" rather than
    reporting a fabricated number.
    """
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            # Explicit argtypes/restype are required: GetCurrentProcess's
            # real return value is a 64-bit pseudo-handle (-1, i.e.
            # 0xFFFFFFFFFFFFFFFF); ctypes' default (32-bit int) marshalling
            # silently truncates/mismatches it on 64-bit Python, causing
            # GetProcessMemoryInfo to fail with ERROR_INVALID_HANDLE.
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            if ok:
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
        except Exception:  # noqa: BLE001 - memory reporting must never break ingestion
            return None
        return None

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KB on Linux, bytes on macOS.
        divisor = 1024 if sys.platform != "darwin" else (1024 * 1024)
        return round(usage / divisor, 1)
    except Exception:  # noqa: BLE001
        return None


def _tags_to_dict(tags: Any) -> dict[str, str]:
    """Eagerly copy an `osmium.osm.TagList` into a plain dict.

    Must happen *inside* the triggering callback: pyosmium's node/way/
    relation objects (and their `.tags`) are only valid for the duration
    of that callback and are invalidated as soon as it returns.
    """
    return {tag.k: tag.v for tag in tags}


@dataclass
class PbfScanStats:
    """Streaming-scan statistics for one `load_osm_pbf` run.

    All counts are exact (every node/way/relation in the file increments
    the corresponding `*_scanned` counter), not estimates or samples.
    """

    input_file: str
    file_size_bytes: int
    nodes_scanned: int = 0
    ways_scanned: int = 0
    relations_scanned: int = 0
    candidate_nodes: int = 0
    candidate_ways: int = 0
    candidate_relations: int = 0
    ways_geometry_unresolved: int = 0
    relations_geometry_unavailable: int = 0
    processing_seconds: float = 0.0
    location_index: str = DEFAULT_LOCATION_INDEX
    peak_memory_mb: float | None = None

    @property
    def osm_objects_scanned(self) -> int:
        return self.nodes_scanned + self.ways_scanned + self.relations_scanned

    @property
    def candidate_objects(self) -> int:
        return self.candidate_nodes + self.candidate_ways + self.candidate_relations

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_file": self.input_file,
            "file_size_bytes": self.file_size_bytes,
            "location_index": self.location_index,
            "osm_objects_scanned": self.osm_objects_scanned,
            "nodes_scanned": self.nodes_scanned,
            "ways_scanned": self.ways_scanned,
            "relations_scanned": self.relations_scanned,
            "candidate_objects": self.candidate_objects,
            "candidate_nodes": self.candidate_nodes,
            "candidate_ways": self.candidate_ways,
            "candidate_relations": self.candidate_relations,
            "ways_geometry_unresolved": self.ways_geometry_unresolved,
            "relations_geometry_unavailable": self.relations_geometry_unavailable,
            "processing_seconds": round(self.processing_seconds, 3),
            "peak_memory_mb": self.peak_memory_mb,
        }


class _CandidateHandler(osmium.SimpleHandler):  # type: ignore[misc]
    """Streaming pyosmium handler: filters candidates, builds geometry eagerly.

    Only matched candidate records (expected to be a small fraction of a
    country's total object count) are appended to `self.records`; every
    other object is rejected at the `_is_relevant_tags` check and
    contributes nothing beyond a counter increment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.nodes_scanned = 0
        self.ways_scanned = 0
        self.relations_scanned = 0
        self.candidate_nodes = 0
        self.candidate_ways = 0
        self.candidate_relations = 0
        self.ways_geometry_unresolved = 0
        self.relations_geometry_unavailable = 0

    def node(self, n: Any) -> None:
        self.nodes_scanned += 1
        if not _is_relevant_tags(n.tags):
            return
        self.candidate_nodes += 1
        tags = _tags_to_dict(n.tags)
        geometry = Point(n.location.lon, n.location.lat) if n.location.valid() else None
        self.records.append(
            {
                "osm_id": str(n.id),
                "osm_type": "node",
                "name": tags.get("name"),
                "raw_tags": tags,
                "geometry": geometry,
            }
        )

    def way(self, w: Any) -> None:
        self.ways_scanned += 1
        if not _is_relevant_tags(w.tags):
            return
        self.candidate_ways += 1
        tags = _tags_to_dict(w.tags)

        coords: list[tuple[float, float]] = []
        resolved = True
        for node_ref in w.nodes:
            if not node_ref.location.valid():
                resolved = False
                break
            coords.append((node_ref.location.lon, node_ref.location.lat))

        geometry: Any = None
        if not resolved or len(coords) < 2:
            self.ways_geometry_unresolved += 1
        elif w.is_closed() and len(coords) >= 4:
            try:
                geometry = Polygon(coords)
            except Exception:  # noqa: BLE001 - malformed ring; leave geometry unavailable
                geometry = None
        else:
            # Open way: LineString is not a supported facility geometry type
            # (see facility_schema.SUPPORTED_GEOMETRY_TYPES). Preserved with
            # this geometry anyway so downstream validation can flag/reject
            # it explicitly (with a reason) rather than this loader silently
            # dropping it.
            try:
                geometry = LineString(coords)
            except Exception:  # noqa: BLE001
                geometry = None

        self.records.append(
            {
                "osm_id": str(w.id),
                "osm_type": "way",
                "name": tags.get("name"),
                "raw_tags": tags,
                "geometry": geometry,
            }
        )

    def relation(self, r: Any) -> None:
        self.relations_scanned += 1
        if not _is_relevant_tags(r.tags):
            return
        self.candidate_relations += 1
        self.relations_geometry_unavailable += 1
        tags = _tags_to_dict(r.tags)
        # KNOWN LIMITATION (documented in module docstring): full
        # multipolygon relation geometry reconstruction is not implemented.
        # The relation's identity/tags are preserved; geometry is left
        # unavailable so it is flagged and preserved (never silently
        # dropped) by the existing, unmodified validation logic.
        self.records.append(
            {
                "osm_id": str(r.id),
                "osm_type": "relation",
                "name": tags.get("name"),
                "raw_tags": tags,
                "geometry": None,
            }
        )


def load_osm_pbf(
    path: str | Path,
    location_index: str = DEFAULT_LOCATION_INDEX,
) -> tuple[gpd.GeoDataFrame, PbfScanStats]:
    """Stream a `.osm.pbf` file, extracting only industrially-relevant candidates.

    Never loads the whole file into memory: pyosmium dispatches one
    node/way/relation at a time to `_CandidateHandler`, which rejects the
    overwhelming majority of objects at a cheap tag check before any
    Python dict or shapely geometry is built. Only matched candidates are
    retained and returned.

    Args:
        path: Path to a `.osm.pbf` file.
        location_index: pyosmium location-index type used to resolve way
            node coordinates (see `DEFAULT_LOCATION_INDEX`). Pass a
            disk-backed type (e.g. ``"dense_file_array,/tmp/idx.store"``)
            to trade speed for lower peak RAM on memory-constrained
            machines.

    Returns:
        A tuple of:

        * A `GeoDataFrame` with the same columns as
          `osm_loader.load_osm_geojson`/`load_osm_csv` (``osm_id``,
          ``osm_type``, ``name``, ``raw_tags``, ``geometry``; CRS
          EPSG:4326) -- ready for `osm_normalization.normalize_osm_facilities`.
        * A `PbfScanStats` with exact scan/candidate/rejection counts and
          timing, for the Stage I.1 report.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    pbf_path = Path(path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"OSM PBF file not found: {pbf_path}")

    file_size_bytes = pbf_path.stat().st_size
    start_time = time.perf_counter()

    handler = _CandidateHandler()
    handler.apply_file(str(pbf_path), locations=True, idx=location_index)

    processing_seconds = time.perf_counter() - start_time
    peak_memory_mb = _peak_working_set_mb()

    stats = PbfScanStats(
        input_file=str(pbf_path),
        file_size_bytes=file_size_bytes,
        nodes_scanned=handler.nodes_scanned,
        ways_scanned=handler.ways_scanned,
        relations_scanned=handler.relations_scanned,
        candidate_nodes=handler.candidate_nodes,
        candidate_ways=handler.candidate_ways,
        candidate_relations=handler.candidate_relations,
        ways_geometry_unresolved=handler.ways_geometry_unresolved,
        relations_geometry_unavailable=handler.relations_geometry_unavailable,
        processing_seconds=processing_seconds,
        location_index=location_index,
        peak_memory_mb=peak_memory_mb,
    )

    if handler.records:
        osm_ids = [rec["osm_id"] for rec in handler.records]
        osm_types = [rec["osm_type"] for rec in handler.records]
        names = [rec["name"] for rec in handler.records]
        raw_tags = [rec["raw_tags"] for rec in handler.records]
        geometries = [rec["geometry"] for rec in handler.records]
    else:
        osm_ids, osm_types, names, raw_tags, geometries = [], [], [], [], []

    result = gpd.GeoDataFrame(
        {
            "osm_id": pd.Series(osm_ids, dtype="object"),
            "osm_type": pd.Series(osm_types, dtype="object"),
            "name": pd.Series(names, dtype="object"),
            "raw_tags": pd.Series(raw_tags, dtype="object"),
            "geometry": geometries,
        },
        crs="EPSG:4326",
    )
    return result, stats
