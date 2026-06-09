"""
Lightweight drainage-network model inspired by SWMM.

This is not a full EPA SWMM replacement. It keeps the same engineering ideas:
nodes, conduits, subcatchments, rainfall-runoff, Manning pipe capacity, routing,
continuity checks, and visual diagnostics. The implementation is intentionally
small and efficient: each time step is O(number_of_subcatchments + number_of_pipes).

Run:
    python3 test.py

Run Python dynamic-wave approximation only:
    python3 test.py --approx-only

Run built-in demo instead of auto-detected shapefiles:
    python3 test.py --demo

Run from shapefiles:
    python3 test.py --from-shp manhole.shp outlet.shp pipe.shp

Run from shapefiles with catchments:
    python3 test.py --from-shp manhole.shp outlet.shp pipe.shp --catchment catchment.shp

Run shapefile topology without synthetic hydrology:
    python3 test.py --from-shp manhole.shp outlet.shp pipe.shp --no-fake-hydrology

Skip PySWMM comparison:
    python3 test.py --no-compare-pyswmm

Route PySWMM node flooding through flooding.py / ANUGA. This is enabled by
default in the script configuration below:
    python3 test.py --couple-surface-flood
    python3 test.py --couple-surface-flood --surface-mesh-size 80 --surface-max-grid-cells 5000 --surface-max-frames 10
    python3 test.py --surface-dem S200R200852100.dem
    python3 test.py --no-surface-dem
    python3 test.py --no-osm-basemap --no-surface-catchments --no-surface-network
    python3 test.py --osm-slider
    python3 test.py --no-couple-surface-flood
    python3 test.py --no-surface-show

Optional plotting dependency:
    python3 -m pip install matplotlib
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
from math import acos, ceil, pi, sin, sqrt
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple


MM_PER_H_TO_M_PER_S = 1.0 / 1000.0 / 3600.0

# Surface-flood coupling defaults. Edit these values when you want normal
# script runs to use a different 2D flood setup.
SURFACE_FLOOD_ENABLED = True
SURFACE_FLOOD_OUTPUT_DIR = "coupled_flood_output"
SURFACE_DEM_PATH = "S200R200852100.dem"
SURFACE_MESH_SIZE_M = 60.0
SURFACE_MAX_GRID_CELLS = 1000
# HTML frame rendering uses every output time step up to this cap.
SURFACE_MAX_FRAMES = 36
SURFACE_SHOW_INTERACTIVE_PLOT = True
SURFACE_SHOW_OSM_BASEMAP = True
SURFACE_SHOW_OSM_IN_SLIDER = False
SURFACE_SHOW_CATCHMENTS = True
SURFACE_SHOW_NETWORK = True
SURFACE_MAP_CRS = "EPSG:2326"
# Leave as None to let Matplotlib choose the available GUI backend.
# Set to "TkAgg", "QtAgg", or "MacOSX" only if you need to force one.
SURFACE_INTERACTIVE_BACKEND = None

# Use SWMM's native hydrology and pollutant buildup/washoff blocks when writing
# the PySWMM input file. This is required for PySWMM-native water quality.
SWMM_NATIVE_WATER_QUALITY_ENABLED = True
SWMM_WQ_POLLUTANTS = [
    {
        "name": "TRACER",
        "units": "MG/L",
        "landuse": "URBAN",
        "buildup_coeff1": 15.0,
        "buildup_coeff2": 1.0,
        "washoff_emc": 35.0,
    },
    {
        "name": "NH4N",
        "units": "MG/L",
        "landuse": "URBAN",
        "buildup_coeff1": 5.0,
        "buildup_coeff2": 1.0,
        "washoff_emc": 4.0,
    },
]

RUNTIME_CACHE_DIR = Path(__file__).resolve().parent / ".runtime_cache"
RUNTIME_CACHE_DIR.mkdir(exist_ok=True)
MPL_CACHE_DIR = RUNTIME_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = RUNTIME_CACHE_DIR / "xdg"
MPL_CACHE_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
if SURFACE_SHOW_INTERACTIVE_PLOT and "--no-surface-show" not in sys.argv:
    if SURFACE_INTERACTIVE_BACKEND:
        os.environ["MPLBACKEND"] = SURFACE_INTERACTIVE_BACKEND
else:
    os.environ.setdefault("MPLBACKEND", "Agg")


class Node:
    """Drainage node/junction/outfall."""

    def __init__(
        self,
        name: str,
        invert_m: float,
        max_depth_m: float,
        x: float,
        y: float,
        is_outfall: bool = False,
    ) -> None:
        self.name = name
        self.invert_m = invert_m
        self.max_depth_m = max_depth_m
        self.x = x
        self.y = y
        self.is_outfall = is_outfall

    @property
    def storage_capacity_m3(self) -> float:
        # Small conceptual node storage. SWMM uses detailed node geometry;
        # this compact model uses storage to represent surcharge buffering.
        return max(self.max_depth_m, 0.05) * 8.0


class Conduit:
    """Circular pipe using Manning full-flow capacity."""

    def __init__(
        self,
        name: str,
        inlet: str,
        outlet: str,
        length_m: float,
        diameter_m: float,
        roughness_n: float = 0.013,
        vertices: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.name = name
        self.inlet = inlet
        self.outlet = outlet
        self.length_m = length_m
        self.diameter_m = diameter_m
        self.roughness_n = roughness_n
        self.vertices = vertices or []

    def slope(self, nodes: Dict[str, Node]) -> float:
        dz = nodes[self.inlet].invert_m - nodes[self.outlet].invert_m
        return max(dz / max(self.length_m, 1e-6), 1e-5)

    def full_area_m2(self) -> float:
        return pi * self.diameter_m**2 / 4.0

    def hydraulic_radius_m(self) -> float:
        return self.diameter_m / 4.0

    def circular_section(self, depth_m: float) -> Tuple[float, float]:
        """Return area and hydraulic radius for a partially full circular pipe."""

        depth = max(0.0, min(depth_m, self.diameter_m))
        if depth <= 1e-6:
            return 0.0, 0.0
        if depth >= self.diameter_m - 1e-6:
            return self.full_area_m2(), self.hydraulic_radius_m()

        radius = self.diameter_m / 2.0
        theta = 2.0 * acos((radius - depth) / radius)
        area = 0.5 * radius * radius * (theta - sin(theta))
        wetted_perimeter = radius * theta
        hydraulic_radius = area / max(wetted_perimeter, 1e-9)
        return area, hydraulic_radius

    def capacity_m3s(self, nodes: Dict[str, Node]) -> float:
        area = self.full_area_m2()
        radius = self.hydraulic_radius_m()
        return (1.0 / self.roughness_n) * area * radius ** (2.0 / 3.0) * sqrt(self.slope(nodes))

    def velocity_mps(self, nodes: Dict[str, Node]) -> float:
        return max(self.capacity_m3s(nodes) / max(self.full_area_m2(), 1e-9), 0.05)


class Subcatchment:
    """Hydrologic area draining to a node."""

    def __init__(
        self,
        name: str,
        outlet: str,
        area_ha: float,
        impervious_fraction: float,
        width_m: float,
        slope: float,
        depression_storage_mm: float = 1.5,
        max_infiltration_mm_h: float = 45.0,
        min_infiltration_mm_h: float = 6.0,
        infiltration_decay_1_h: float = 3.5,
    ) -> None:
        self.name = name
        self.outlet = outlet
        self.area_ha = area_ha
        self.impervious_fraction = impervious_fraction
        self.width_m = width_m
        self.slope = slope
        self.depression_storage_mm = depression_storage_mm
        self.max_infiltration_mm_h = max_infiltration_mm_h
        self.min_infiltration_mm_h = min_infiltration_mm_h
        self.infiltration_decay_1_h = infiltration_decay_1_h

    @property
    def area_m2(self) -> float:
        return self.area_ha * 10000.0

    def runoff_m3s(self, rain_mm_h: float, elapsed_h: float) -> float:
        """Simple SWMM-like runoff using Horton-style infiltration and runoff coefficient.

        Impervious area routes most rainfall to runoff after depression storage.
        Pervious area loses rainfall to a decaying infiltration capacity.
        """

        if rain_mm_h <= 0.0:
            return 0.0

        # Horton-style infiltration capacity.
        infil_capacity = self.min_infiltration_mm_h + (
            self.max_infiltration_mm_h - self.min_infiltration_mm_h
        ) * pow(2.718281828, -self.infiltration_decay_1_h * elapsed_h)

        effective_imp = max(rain_mm_h - self.depression_storage_mm, 0.0)
        effective_perv = max(rain_mm_h - infil_capacity, 0.0)

        runoff_mm_h = (
            self.impervious_fraction * effective_imp
            + (1.0 - self.impervious_fraction) * effective_perv
        )

        # Wider/steeper catchments respond faster. This is a compact response
        # factor, not a full nonlinear SWMM overland-flow solution.
        response = min(1.0, 0.35 + 0.015 * self.width_m * sqrt(max(self.slope, 1e-4)))
        return runoff_mm_h * MM_PER_H_TO_M_PER_S * self.area_m2 * response


class SimulationResult(NamedTuple):
    time_min: List[float]
    rain_mm_h: List[float]
    node_inflow_m3s: Dict[str, List[float]]
    node_surcharge_m3: Dict[str, List[float]]
    node_flooding_m3s: Dict[str, List[float]]
    node_pollutant_conc: Dict[str, Dict[str, List[float]]]
    conduit_flow_m3s: Dict[str, List[float]]
    outfall_flow_m3s: Dict[str, List[float]]
    flooding_m3: Dict[str, float]
    continuity_error_pct: float


class PyswmmComparison(NamedTuple):
    inp_path: str
    link_flow_m3s: Dict[str, List[float]]
    node_flooding_m3s: Dict[str, List[float]]
    node_pollutant_conc: Dict[str, Dict[str, List[float]]]
    summary_rows: List[Dict[str, float]]


class DrainageNetwork:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.conduits: Dict[str, Conduit] = {}
        self.subcatchments: Dict[str, Subcatchment] = {}
        self.catchment_polygons: List[List[Tuple[float, float]]] = []
        self._out_edges: Dict[str, List[str]] = defaultdict(list)
        self._in_edges: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_conduit(self, conduit: Conduit) -> None:
        if conduit.inlet not in self.nodes or conduit.outlet not in self.nodes:
            raise ValueError(f"Conduit {conduit.name} references a missing node")
        self.conduits[conduit.name] = conduit
        self._out_edges[conduit.inlet].append(conduit.name)
        self._in_edges[conduit.outlet].append(conduit.name)

    def add_subcatchment(self, subcatchment: Subcatchment) -> None:
        if subcatchment.outlet not in self.nodes:
            raise ValueError(f"Subcatchment {subcatchment.name} has a missing outlet node")
        self.subcatchments[subcatchment.name] = subcatchment

    def topological_order(self) -> List[str]:
        indegree = {name: len(self._in_edges[name]) for name in self.nodes}
        queue = deque([name for name, degree in indegree.items() if degree == 0])
        order: List[str] = []

        while queue:
            node_name = queue.popleft()
            order.append(node_name)
            for edge_name in self._out_edges[node_name]:
                outlet = self.conduits[edge_name].outlet
                indegree[outlet] -= 1
                if indegree[outlet] == 0:
                    queue.append(outlet)

        if len(order) != len(self.nodes):
            raise ValueError("The simplified routing model requires an acyclic network")
        return order

    def simulate(
        self,
        rainfall_mm_h: Iterable[float],
        dt_s: float = 300.0,
    ) -> SimulationResult:
        """Run event simulation.

        Routing assumptions:
        - Subcatchments add runoff to outlet nodes.
        - Each node sends water to downstream conduits up to pipe capacity.
        - Pipe travel time is represented with fixed delay queues.
        - Excess node storage becomes surcharge; overflow above node capacity is flooding.
        """

        rain = list(rainfall_mm_h)
        if not rain:
            raise ValueError("rainfall_mm_h cannot be empty")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        order = self.topological_order()
        capacities = {name: conduit.capacity_m3s(self.nodes) for name, conduit in self.conduits.items()}
        delay_steps = {
            name: max(1, int(ceil(conduit.length_m / conduit.velocity_mps(self.nodes) / dt_s)))
            for name, conduit in self.conduits.items()
        }
        pipe_queues = {
            name: deque([0.0] * delay_steps[name], maxlen=delay_steps[name])
            for name in self.conduits
        }

        stored = {name: 0.0 for name in self.nodes}
        flooding = {name: 0.0 for name in self.nodes}

        node_inflow = {name: [] for name in self.nodes}
        node_surcharge = {name: [] for name in self.nodes}
        node_flooding = {name: [] for name in self.nodes}
        conduit_flow = {name: [] for name in self.conduits}
        outfall_flow = {name: [] for name, node in self.nodes.items() if node.is_outfall}

        total_rain_runoff_m3 = 0.0
        total_outfall_m3 = 0.0

        for step, rain_mm_h in enumerate(rain):
            elapsed_h = step * dt_s / 3600.0
            arriving = {name: 0.0 for name in self.nodes}

            # Release delayed pipe flow to receiving nodes.
            for conduit_name, q in pipe_queues.items():
                released_m3 = q.popleft()
                arriving[self.conduits[conduit_name].outlet] += released_m3
                q.append(0.0)

            # Hydrologic runoff.
            for subcatchment in self.subcatchments.values():
                runoff_q = subcatchment.runoff_m3s(rain_mm_h, elapsed_h)
                runoff_m3 = runoff_q * dt_s
                arriving[subcatchment.outlet] += runoff_m3
                total_rain_runoff_m3 += runoff_m3

            outgoing_q_this_step = {name: 0.0 for name in self.conduits}
            outfall_q_this_step = {name: 0.0 for name in outfall_flow}
            flooding_q_this_step = {name: 0.0 for name in self.nodes}

            for node_name in order:
                node = self.nodes[node_name]
                stored[node_name] += arriving[node_name]

                if node.is_outfall:
                    outfall_q = stored[node_name] / dt_s
                    outfall_q_this_step[node_name] = outfall_q
                    total_outfall_m3 += stored[node_name]
                    stored[node_name] = 0.0
                    continue

                outgoing_edges = self._out_edges[node_name]
                if outgoing_edges:
                    total_capacity = sum(capacities[edge] for edge in outgoing_edges)
                    send_q = min(stored[node_name] / dt_s, total_capacity)
                    send_m3 = send_q * dt_s

                    # Split by capacity share when there are parallel/downstream branches.
                    for edge in outgoing_edges:
                        share = capacities[edge] / total_capacity if total_capacity > 0.0 else 0.0
                        flow_m3 = send_m3 * share
                        pipe_queues[edge][-1] += flow_m3
                        outgoing_q_this_step[edge] = flow_m3 / dt_s
                    stored[node_name] -= send_m3

                capacity = node.storage_capacity_m3
                if stored[node_name] > capacity:
                    overflow = stored[node_name] - capacity
                    flooding[node_name] += overflow
                    flooding_q_this_step[node_name] = overflow / dt_s
                    stored[node_name] = capacity

            for name in self.nodes:
                node_inflow[name].append(arriving[name] / dt_s)
                node_surcharge[name].append(stored[name])
                node_flooding[name].append(flooding_q_this_step[name])
            for name in self.conduits:
                conduit_flow[name].append(outgoing_q_this_step[name])
            for name in outfall_flow:
                outfall_flow[name].append(outfall_q_this_step[name])

        remaining_storage = sum(stored.values()) + sum(sum(q) for q in pipe_queues.values())
        total_flooding = sum(flooding.values())
        denominator = max(total_rain_runoff_m3, 1e-9)
        continuity_error_pct = 100.0 * (
            total_rain_runoff_m3 - total_outfall_m3 - total_flooding - remaining_storage
        ) / denominator

        return SimulationResult(
            time_min=[i * dt_s / 60.0 for i in range(len(rain))],
            rain_mm_h=rain,
            node_inflow_m3s=node_inflow,
            node_surcharge_m3=node_surcharge,
            node_flooding_m3s=node_flooding,
            node_pollutant_conc={},
            conduit_flow_m3s=conduit_flow,
            outfall_flow_m3s=outfall_flow,
            flooding_m3=flooding,
            continuity_error_pct=continuity_error_pct,
        )

    def simulate_dynamic_wave(
        self,
        rainfall_mm_h: Iterable[float],
        report_dt_s: float = 300.0,
        routing_dt_s: float = 30.0,
    ) -> SimulationResult:
        """Run a compact dynamic-wave-like hydraulic simulation.

        This is still a reduced model, but it is much closer to SWMM DYNWAVE
        than the capacity-delay router: each internal routing step computes
        conduit flow from hydraulic-grade difference between connected nodes,
        allows branching to emerge from local heads, and relaxes pipe flow over
        time rather than instantly splitting by capacity.
        """

        rain = list(rainfall_mm_h)
        if not rain:
            raise ValueError("rainfall_mm_h cannot be empty")
        if report_dt_s <= 0.0 or routing_dt_s <= 0.0:
            raise ValueError("time steps must be positive")

        substeps = max(1, int(ceil(report_dt_s / routing_dt_s)))
        hdt = report_dt_s / substeps
        storage_area = {
            name: max(node.storage_capacity_m3 / max(node.max_depth_m, 0.1), 1.0)
            for name, node in self.nodes.items()
        }
        for conduit in self.conduits.values():
            pipe_storage_area = conduit.full_area_m2() * conduit.length_m / max(conduit.diameter_m, 0.1)
            storage_area[conduit.inlet] += 0.05 * pipe_storage_area
            storage_area[conduit.outlet] += 0.05 * pipe_storage_area
        node_volume = {name: 0.0 for name in self.nodes}
        pipe_flow = {name: 0.0 for name in self.conduits}
        flooding = {name: 0.0 for name in self.nodes}

        node_inflow = {name: [] for name in self.nodes}
        node_surcharge = {name: [] for name in self.nodes}
        node_flooding = {name: [] for name in self.nodes}
        conduit_flow = {name: [] for name in self.conduits}
        outfall_flow = {name: [] for name, node in self.nodes.items() if node.is_outfall}

        total_runoff_m3 = 0.0
        total_outfall_m3 = 0.0

        for report_step, rain_mm_h in enumerate(rain):
            report_node_inflow = {name: 0.0 for name in self.nodes}
            report_node_flooding = {name: 0.0 for name in self.nodes}
            report_conduit_q = {name: 0.0 for name in self.conduits}
            report_outfall_q = {name: 0.0 for name in outfall_flow}

            for substep in range(substeps):
                elapsed_h = (report_step * report_dt_s + substep * hdt) / 3600.0
                external_inflow = {name: 0.0 for name in self.nodes}

                for subcatchment in self.subcatchments.values():
                    q = subcatchment.runoff_m3s(rain_mm_h, elapsed_h)
                    volume = q * hdt
                    node_volume[subcatchment.outlet] += volume
                    external_inflow[subcatchment.outlet] += q
                    total_runoff_m3 += volume

                heads = {}
                for name, node in self.nodes.items():
                    if node.is_outfall:
                        heads[name] = node.invert_m
                    else:
                        heads[name] = node.invert_m + max(node_volume[name] / storage_area[name], 0.0)

                volume_delta = {name: 0.0 for name in self.nodes}
                outfall_volume = {name: 0.0 for name in outfall_flow}

                for conduit_name, conduit in self.conduits.items():
                    h1 = heads[conduit.inlet]
                    h2 = heads[conduit.outlet]
                    dh = h1 - h2
                    if abs(dh) < 1e-9:
                        target_q = 0.0
                    else:
                        source_node = self.nodes[conduit.inlet] if dh > 0.0 else self.nodes[conduit.outlet]
                        source_depth = max((h1 if dh > 0.0 else h2) - source_node.invert_m, 0.0)
                        area, radius = conduit.circular_section(source_depth)
                        if area <= 0.0 or radius <= 0.0:
                            target_q = 0.0
                            pipe_flow[conduit_name] = 0.50 * pipe_flow[conduit_name]
                            report_conduit_q[conduit_name] += pipe_flow[conduit_name]
                            continue
                        hydraulic_slope = abs(dh) / max(conduit.length_m, 1e-6)
                        target_q = (1.0 / conduit.roughness_n) * area * radius ** (2.0 / 3.0) * sqrt(hydraulic_slope)
                        if dh < 0.0:
                            target_q = -target_q

                    # Free outfalls should not push water back into the network.
                    if self.nodes[conduit.outlet].is_outfall and target_q < 0.0:
                        target_q = 0.0
                    if self.nodes[conduit.inlet].is_outfall and target_q > 0.0:
                        target_q = 0.0

                    q = 0.70 * pipe_flow[conduit_name] + 0.30 * target_q

                    source = conduit.inlet if q >= 0.0 else conduit.outlet
                    if not self.nodes[source].is_outfall:
                        max_available_q = max(node_volume[source] + volume_delta[source], 0.0) / hdt
                        if abs(q) > max_available_q:
                            q = max_available_q if q >= 0.0 else -max_available_q

                    pipe_flow[conduit_name] = q
                    moved = q * hdt
                    volume_delta[conduit.inlet] -= moved
                    volume_delta[conduit.outlet] += moved
                    report_conduit_q[conduit_name] += q

                for name, delta in volume_delta.items():
                    node = self.nodes[name]
                    if node.is_outfall:
                        if delta > 0.0:
                            outfall_volume[name] += delta
                            total_outfall_m3 += delta
                        node_volume[name] = 0.0
                    else:
                        node_volume[name] = max(node_volume[name] + delta, 0.0)
                        max_storage = node.storage_capacity_m3
                        if node_volume[name] > max_storage:
                            overflow = node_volume[name] - max_storage
                            flooding[name] += overflow
                            report_node_flooding[name] += overflow / hdt
                            node_volume[name] = max_storage

                for name, q in external_inflow.items():
                    report_node_inflow[name] += q
                for name, volume in outfall_volume.items():
                    report_outfall_q[name] += volume / hdt

            for name in self.nodes:
                node_inflow[name].append(report_node_inflow[name] / substeps)
                node_surcharge[name].append(node_volume[name])
                node_flooding[name].append(report_node_flooding[name] / substeps)
            for name in self.conduits:
                conduit_flow[name].append(report_conduit_q[name] / substeps)
            for name in outfall_flow:
                outfall_flow[name].append(report_outfall_q[name] / substeps)

        remaining_storage = sum(node_volume.values())
        total_flooding = sum(flooding.values())
        continuity_error_pct = 100.0 * (
            total_runoff_m3 - total_outfall_m3 - total_flooding - remaining_storage
        ) / max(total_runoff_m3, 1e-9)

        return SimulationResult(
            time_min=[i * report_dt_s / 60.0 for i in range(len(rain))],
            rain_mm_h=rain,
            node_inflow_m3s=node_inflow,
            node_surcharge_m3=node_surcharge,
            node_flooding_m3s=node_flooding,
            node_pollutant_conc={},
            conduit_flow_m3s=conduit_flow,
            outfall_flow_m3s=outfall_flow,
            flooding_m3=flooding,
            continuity_error_pct=continuity_error_pct,
        )

    def write_swmm_inp(
        self,
        path: str,
        node_runoff_m3s: Optional[Dict[str, List[float]]] = None,
        rainfall_mm_h: Optional[Iterable[float]] = None,
        dt_s: float = 300.0,
        use_native_hydrology: bool = False,
        pollutants: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Export a runnable SWMM-compatible INP file.

        If node_runoff_m3s is provided, it is written as external FLOW inflow
        time series. That lets the same INP be used by PySWMM for comparison.
        """

        node_runoff_m3s = node_runoff_m3s or {}
        rainfall = list(rainfall_mm_h or [])
        pollutants = pollutants or []
        n_steps = max(
            (len(series) for series in node_runoff_m3s.values()),
            default=len(rainfall) if rainfall else 36,
        )
        duration = timedelta(seconds=dt_s * n_steps)
        start = datetime(2026, 1, 1, 0, 0, 0)
        end = start + duration
        report_step_min = int(dt_s // 60)
        report_step_sec = int(dt_s % 60)
        swmm_routing_step_s = min(int(dt_s), 30)
        routing_step_min = swmm_routing_step_s // 60
        routing_step_sec = swmm_routing_step_s % 60

        lines = [
            "[TITLE]",
            ";; Lightweight Python drainage model exported as a SWMM-compatible INP",
            "",
            "[OPTIONS]",
            "FLOW_UNITS           CMS",
            "INFILTRATION         HORTON",
            "FLOW_ROUTING         DYNWAVE",
            "LINK_OFFSETS         DEPTH",
            "MIN_SLOPE            0.00001",
            "ALLOW_PONDING        NO",
            "IGNORE_QUALITY        NO",
            f"START_DATE           {start:%m/%d/%Y}",
            f"START_TIME           {start:%H:%M:%S}",
            f"REPORT_START_DATE    {start:%m/%d/%Y}",
            f"REPORT_START_TIME    {start:%H:%M:%S}",
            f"END_DATE             {end:%m/%d/%Y}",
            f"END_TIME             {end:%H:%M:%S}",
            f"REPORT_STEP          00:{report_step_min:02d}:{report_step_sec:02d}",
            f"ROUTING_STEP         00:{routing_step_min:02d}:{routing_step_sec:02d}",
            "",
            "[JUNCTIONS]",
            ";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded",
        ]

        for node in self.nodes.values():
            if not node.is_outfall:
                lines.append(
                    f"{node.name:<16} {node.invert_m:<10.3f} {node.max_depth_m:<10.3f} "
                    "0.000      0.000      0.000"
                )

        lines.extend(["", "[OUTFALLS]", ";;Name           Elevation  Type       Stage Data  Gated"])
        for node in self.nodes.values():
            if node.is_outfall:
                lines.append(f"{node.name:<16} {node.invert_m:<10.3f} FREE       NO")

        lines.extend(
            [
                "",
                "[CONDUITS]",
                ";;Name           FromNode         ToNode           Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow",
            ]
        )
        for conduit in self.conduits.values():
            lines.append(
                f"{conduit.name:<16} {conduit.inlet:<16} {conduit.outlet:<16} "
                f"{conduit.length_m:<10.2f} {conduit.roughness_n:<10.4f} "
                "0.000      0.000      0.000      0.000"
            )

        lines.extend(["", "[XSECTIONS]", ";;Link           Shape      Geom1      Geom2  Geom3  Geom4  Barrels"])
        for conduit in self.conduits.values():
            lines.append(f"{conduit.name:<16} CIRCULAR   {conduit.diameter_m:<10.3f} 0      0      0      1")

        if use_native_hydrology:
            lines.extend(["", "[RAINGAGES]", ";;Name           Format     Interval  SCF      Source"])
            report_step_hhmm = f"{int(dt_s // 3600)}:{int((dt_s % 3600) // 60):02d}"
            lines.append(f"RG1              INTENSITY  {report_step_hhmm:<8} 1.0      TIMESERIES Rainfall")

            lines.extend(
                [
                    "",
                    "[SUBCATCHMENTS]",
                    ";;Name           RainGage   Outlet           Area      %Imperv  Width     %Slope    CurbLen",
                ]
            )
            for subcatchment in self.subcatchments.values():
                lines.append(
                    f"{subcatchment.name:<16} RG1        {subcatchment.outlet:<16} "
                    f"{subcatchment.area_ha:<9.4f} {100.0 * subcatchment.impervious_fraction:<8.2f} "
                    f"{subcatchment.width_m:<9.2f} {100.0 * subcatchment.slope:<8.3f} 0"
                )

            lines.extend(
                [
                    "",
                    "[SUBAREAS]",
                    ";;Subcatchment   N-Imperv  N-Perv    S-Imperv  S-Perv    PctZero   RouteTo   PctRouted",
                ]
            )
            for subcatchment in self.subcatchments.values():
                lines.append(
                    f"{subcatchment.name:<16} 0.015     0.15      "
                    f"{subcatchment.depression_storage_mm:<9.3f} {subcatchment.depression_storage_mm:<9.3f} "
                    "25        OUTLET"
                )

            lines.extend(
                [
                    "",
                    "[INFILTRATION]",
                    ";;Subcatchment   MaxRate   MinRate   Decay     DryTime   MaxInfil",
                ]
            )
            for subcatchment in self.subcatchments.values():
                lines.append(
                    f"{subcatchment.name:<16} {subcatchment.max_infiltration_mm_h:<9.3f} "
                    f"{subcatchment.min_infiltration_mm_h:<9.3f} "
                    f"{subcatchment.infiltration_decay_1_h:<9.3f} 7.0      0.0"
                )

            if pollutants:
                lines.extend(
                    [
                        "",
                        "[POLLUTANTS]",
                        ";;Name           Units  Crain  Cgw    Crdii  Kdecay  SnowOnly  CoPollut  CoFrac  Cdwf   Cinit",
                    ]
                )
                for pollutant in pollutants:
                    lines.append(
                        f"{pollutant['name']:<16} {pollutant.get('units', 'MG/L'):<6} "
                        "0      0      0      0       NO        *         0       0      0"
                    )

                landuses = sorted({pollutant.get("landuse", "URBAN") for pollutant in pollutants})
                lines.extend(["", "[LANDUSES]", ";;Name           SweepingInterval  Availability  LastSwept"])
                for landuse in landuses:
                    lines.append(f"{landuse:<16} 0                 0             0")

                lines.extend(["", "[COVERAGES]", ";;Subcatchment   LandUse          Percent"])
                for subcatchment in self.subcatchments.values():
                    for landuse in landuses:
                        lines.append(f"{subcatchment.name:<16} {landuse:<16} 100")

                lines.extend(["", "[BUILDUP]", ";;LandUse        Pollutant        Function  Coeff1   Coeff2   Coeff3   PerUnit"])
                for pollutant in pollutants:
                    lines.append(
                        f"{pollutant.get('landuse', 'URBAN'):<16} {pollutant['name']:<16} POW       "
                        f"{pollutant.get('buildup_coeff1', 10.0):<8.3f} "
                        f"{pollutant.get('buildup_coeff2', 1.0):<8.3f} 0        AREA"
                    )

                lines.extend(["", "[WASHOFF]", ";;LandUse        Pollutant        Function  Coeff1   Coeff2   SweepRmvl  BmpRmvl"])
                for pollutant in pollutants:
                    lines.append(
                        f"{pollutant.get('landuse', 'URBAN'):<16} {pollutant['name']:<16} EMC       "
                        f"{pollutant.get('washoff_emc', 10.0):<8.3f} 0        0          0"
                    )

        active_inflows = {
            node_name: series
            for node_name, series in node_runoff_m3s.items()
            if not use_native_hydrology and series and max(series) > 0.0
        }

        if active_inflows:
            lines.extend(["", "[INFLOWS]", ";;Node           Parameter  TimeSeries       Type    Mfactor  Sfactor  Baseline"])
            for node_name in active_inflows:
                lines.append(f"{node_name:<16} FLOW       Qin_{node_name:<12} FLOW    1.0      1.0      0.0")

            lines.extend(["", "[TIMESERIES]", ";;Name           Date        Time      Value"])
            for node_name, series in active_inflows.items():
                series_name = f"Qin_{node_name}"
                for step, value in enumerate(series):
                    stamp = start + timedelta(seconds=dt_s * step)
                    lines.append(f"{series_name:<16} {stamp:%m/%d/%Y}  {stamp:%H:%M:%S}  {value:.8f}")
                stamp = start + timedelta(seconds=dt_s * len(series))
                lines.append(f"{series_name:<16} {stamp:%m/%d/%Y}  {stamp:%H:%M:%S}  0.00000000")

        if use_native_hydrology and rainfall:
            lines.extend(["", "[TIMESERIES]", ";;Name           Date        Time      Value"])
            for step, value in enumerate(rainfall):
                stamp = start + timedelta(seconds=dt_s * step)
                lines.append(f"Rainfall         {stamp:%m/%d/%Y}  {stamp:%H:%M:%S}  {value:.8f}")
            stamp = start + timedelta(seconds=dt_s * len(rainfall))
            lines.append(f"Rainfall         {stamp:%m/%d/%Y}  {stamp:%H:%M:%S}  0.00000000")

        lines.extend(["", "[COORDINATES]", ";;Node           X          Y"])
        for node in self.nodes.values():
            lines.append(f"{node.name:<16} {node.x:<10.2f} {node.y:<10.2f}")

        lines.extend(
            [
                "",
                "[REPORT]",
                "INPUT      NO",
                "CONTROLS   NO",
                "SUBCATCHMENTS ALL",
                "NODES ALL",
                "LINKS ALL",
                "",
            ]
        )

        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def local_runoff_by_node(
    model: DrainageNetwork,
    rainfall_mm_h: Iterable[float],
    dt_s: float,
) -> Dict[str, List[float]]:
    """Compute only subcatchment runoff entering each node, excluding pipe arrivals."""

    rain = list(rainfall_mm_h)
    runoff = {name: [0.0] * len(rain) for name in model.nodes}

    for step, rain_mm_h in enumerate(rain):
        elapsed_h = step * dt_s / 3600.0
        for subcatchment in model.subcatchments.values():
            runoff[subcatchment.outlet][step] += subcatchment.runoff_m3s(rain_mm_h, elapsed_h)

    return runoff


def compare_with_pyswmm(
    model: DrainageNetwork,
    python_result: SimulationResult,
    dt_s: float = 300.0,
    inp_path: str = "demo_network.inp",
) -> Optional[PyswmmComparison]:
    """Run the same node inflows through PySWMM and compare pipe-flow statistics.

    PySWMM loads a native SWMM dynamic library. On some macOS setups an ABI or
    architecture mismatch can kill the whole Python process, so the actual
    PySWMM import/run is isolated in a child process.
    """

    local_worker = Path(__file__).resolve().parent / ".venv_pyswmm_test" / "bin" / "python"
    worker_python = str(local_worker) if local_worker.exists() else sys.executable
    cmd = [
        worker_python,
        str(Path(__file__).resolve()),
        "--pyswmm-worker",
        inp_path,
        str(dt_s),
        "--links",
    ] + list(model.conduits)
    cmd += ["--nodes"] + list(model.nodes)
    cmd += ["--pollutants"] + [pollutant["name"] for pollutant in SWMM_WQ_POLLUTANTS]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print(f"PySWMM comparison timed out after writing {inp_path}")
        return None

    if completed.returncode != 0:
        if completed.returncode < 0:
            signal_no = -completed.returncode
            print(f"PySWMM comparison process was killed by signal {signal_no}; input file: {inp_path}")
        else:
            print(f"PySWMM comparison failed while reading {inp_path}")
        if completed.stderr.strip():
            print(completed.stderr.strip())
        elif completed.stdout.strip():
            print(completed.stdout.strip())
        return None

    try:
        worker_result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"PySWMM comparison returned invalid output while reading {inp_path}: {exc}")
        if completed.stdout.strip():
            print(completed.stdout.strip())
        return None

    if isinstance(worker_result, dict) and "links" in worker_result:
        swmm_flows = worker_result.get("links", {})
        node_flooding = worker_result.get("node_flooding_m3s", {})
        node_pollutant_conc = worker_result.get("node_pollutant_conc", {})
    else:
        swmm_flows = worker_result
        node_flooding = {}
        node_pollutant_conc = {}

    expected_len = len(python_result.time_min)
    for name in model.conduits:
        series = list(swmm_flows.get(name, []))
        if len(series) < expected_len:
            fill = series[-1] if series else 0.0
            series.extend([fill] * (expected_len - len(series)))
        elif len(series) > expected_len:
            series = series[:expected_len]
        swmm_flows[name] = series

    for name in model.nodes:
        series = list(node_flooding.get(name, []))
        if len(series) < expected_len:
            series.extend([0.0] * (expected_len - len(series)))
        elif len(series) > expected_len:
            series = series[:expected_len]
        node_flooding[name] = series

        pollutant_data = node_pollutant_conc.get(name, {})
        node_pollutant_conc[name] = {}
        for pollutant in SWMM_WQ_POLLUTANTS:
            pollutant_name = pollutant["name"]
            pollutant_series = list(pollutant_data.get(pollutant_name, []))
            if len(pollutant_series) < expected_len:
                pollutant_series.extend([0.0] * (expected_len - len(pollutant_series)))
            elif len(pollutant_series) > expected_len:
                pollutant_series = pollutant_series[:expected_len]
            node_pollutant_conc[name][pollutant_name] = pollutant_series

    rows: List[Dict[str, float]] = []
    for name in model.conduits:
        py_series = python_result.conduit_flow_m3s[name]
        swmm_series = swmm_flows[name]
        py_peak = max(py_series) if py_series else 0.0
        swmm_peak = max(swmm_series) if swmm_series else 0.0
        py_mean = sum(py_series) / len(py_series) if py_series else 0.0
        swmm_mean = sum(swmm_series) / len(swmm_series) if swmm_series else 0.0
        rows.append(
            {
                "python_peak": py_peak,
                "pyswmm_peak": swmm_peak,
                "peak_diff": swmm_peak - py_peak,
                "python_mean": py_mean,
                "pyswmm_mean": swmm_mean,
                "mean_diff": swmm_mean - py_mean,
            }
        )

    return PyswmmComparison(
        inp_path=inp_path,
        link_flow_m3s=swmm_flows,
        node_flooding_m3s=node_flooding,
        node_pollutant_conc=node_pollutant_conc,
        summary_rows=rows,
    )


def ensure_swmm_toolkit_codesigned() -> bool:
    """Ad-hoc sign bundled macOS dylibs when swmm-toolkit wheel signatures are invalid."""

    if sys.platform != "darwin":
        return True

    site_roots = {
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
        *[p for p in sys.path if "site-packages" in p],
    }
    dylibs = ("libomp.dylib", "libswmm-output.dylib")

    for root in sorted(filter(None, site_roots)):
        toolkit_dir = Path(root) / "swmm" / "toolkit"
        if not toolkit_dir.exists():
            continue
        for name in dylibs:
            dylib = toolkit_dir / name
            if not dylib.exists():
                continue
            verify = subprocess.run(["codesign", "--verify", str(dylib)], capture_output=True, text=True)
            if verify.returncode == 0:
                continue
            sign = subprocess.run(["codesign", "--force", "--sign", "-", str(dylib)], capture_output=True, text=True)
            if sign.returncode != 0:
                print(f"Cannot sign {dylib}: {sign.stderr.strip()}", file=sys.stderr)
                return False
    return True


def pyswmm_worker(argv: List[str]) -> int:
    if len(argv) < 7 or "--links" not in argv or "--nodes" not in argv:
        print(
            "Usage: test.py --pyswmm-worker INP_PATH DT_S --links LINK_NAME... --nodes NODE_NAME... "
            "--pollutants POLLUTANT_NAME...",
            file=sys.stderr,
        )
        return 2

    inp_path = argv[2]
    dt_s = float(argv[3])
    links_idx = argv.index("--links")
    nodes_idx = argv.index("--nodes")
    pollutants_idx = argv.index("--pollutants") if "--pollutants" in argv else len(argv)
    link_names = argv[links_idx + 1 : nodes_idx]
    node_names = argv[nodes_idx + 1 : pollutants_idx]
    pollutant_names = argv[pollutants_idx + 1 :] if pollutants_idx < len(argv) else []

    if not ensure_swmm_toolkit_codesigned():
        return 1

    try:
        from pyswmm import Links, Nodes, Simulation
    except ModuleNotFoundError:
        print("PySWMM is not installed. Install it with: python3 -m pip install pyswmm", file=sys.stderr)
        return 1

    swmm_flows = {name: [] for name in link_names}
    node_flooding = {name: [] for name in node_names}
    node_pollutant_conc = {
        name: {pollutant_name: [] for pollutant_name in pollutant_names}
        for name in node_names
    }

    try:
        with Simulation(inp_path) as sim:
            sim.step_advance(int(dt_s))
            links = Links(sim)
            nodes = Nodes(sim)
            swmm_links = {name: links[name] for name in link_names}
            swmm_nodes = {name: nodes[name] for name in node_names}
            for _ in sim:
                for name, link in swmm_links.items():
                    swmm_flows[name].append(max(float(link.flow), 0.0))
                for name, node in swmm_nodes.items():
                    flooding_rate = getattr(node, "flooding", 0.0)
                    if callable(flooding_rate):
                        flooding_rate = flooding_rate()
                    node_flooding[name].append(max(float(flooding_rate), 0.0))
                    qualities = getattr(node, "pollut_quality", {}) or {}
                    for pollutant_name in pollutant_names:
                        node_pollutant_conc[name][pollutant_name].append(
                            max(float(qualities.get(pollutant_name, 0.0)), 0.0)
                        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "links": swmm_flows,
                "node_flooding_m3s": node_flooding,
                "node_pollutant_conc": node_pollutant_conc,
            }
        )
    )
    return 0


def plot_network(
    model: DrainageNetwork,
    result: Optional[SimulationResult] = None,
    save_path: Optional[str] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed. Install it with: python3 -m pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Drainage network ({len(model.nodes)} nodes, {len(model.conduits)} pipes)")
    ax.set_aspect("equal", adjustable="box")

    if model.catchment_polygons:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        patches = [Polygon(points, closed=True) for points in model.catchment_polygons if len(points) >= 3]
        catchments = PatchCollection(
            patches,
            facecolor="#f2d16b",
            edgecolor="#9a8f62",
            linewidth=0.45,
            alpha=0.28,
            zorder=0,
        )
        ax.add_collection(catchments)

    max_flow = 0.0
    if result:
        max_flow = max((max(v) for v in result.conduit_flow_m3s.values()), default=0.0)

    for conduit in model.conduits.values():
        n1 = model.nodes[conduit.inlet]
        n2 = model.nodes[conduit.outlet]
        vertices = conduit.vertices or [(n1.x, n1.y), (n2.x, n2.y)]
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        mean_flow = 0.0
        if result:
            series = result.conduit_flow_m3s[conduit.name]
            mean_flow = sum(series) / max(len(series), 1)
        width = 1.5 if max_flow <= 0.0 else 1.0 + 5.0 * mean_flow / max_flow
        ax.plot(xs, ys, color="#2f6f8f", linewidth=width, zorder=1)
        if len(model.conduits) <= 40:
            mid = len(vertices) // 2
            ax.text(vertices[mid][0], vertices[mid][1], conduit.name, fontsize=8, color="#1b4d63")

    for node in model.nodes.values():
        marker = "s" if node.is_outfall else "o"
        color = "#2a9d8f" if node.is_outfall else "#e76f51"
        size = 65 if len(model.nodes) <= 60 else (46 if node.is_outfall else 30)
        ax.scatter(node.x, node.y, s=size, marker=marker, color=color, edgecolor="black", linewidth=0.45, zorder=3)
        if node.is_outfall or len(model.nodes) <= 60:
            ax.text(node.x + 8, node.y + 8, node.name, fontsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.scatter([], [], s=30, marker="o", color="#e76f51", edgecolor="black", label="Manhole")
    ax.scatter([], [], s=46, marker="s", color="#2a9d8f", edgecolor="black", label="Outlet")
    ax.plot([], [], color="#2f6f8f", label="Pipe")
    if model.catchment_polygons:
        ax.fill([], [], color="#f2d16b", alpha=0.28, label="Catchment")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"Network plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_results(result: SimulationResult, save_path: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed. Install it with: python3 -m pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    time = result.time_min

    axes[0].bar(time, result.rain_mm_h, width=max(time[1] - time[0], 1.0) * 0.85 if len(time) > 1 else 1)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Rain\n(mm/h)")
    axes[0].set_title("Storm response")

    plotted_flow = False
    flow_names = sorted(result.conduit_flow_m3s, key=lambda name: max(result.conduit_flow_m3s[name]), reverse=True)
    if len(flow_names) > 12:
        flow_names = flow_names[:12]
    for name in flow_names:
        series = result.conduit_flow_m3s[name]
        if max(series) > 0.0 or len(result.conduit_flow_m3s) <= 12:
            axes[1].plot(time, series, label=name)
            plotted_flow = True
    if not plotted_flow:
        axes[1].text(0.5, 0.5, "No conduit flow: missing subcatchment or inflow data", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_ylabel("Pipe flow\n(m3/s)")
    if plotted_flow:
        axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.25)

    plotted_storage = False
    node_names = sorted(result.node_surcharge_m3, key=lambda name: max(result.node_surcharge_m3[name]), reverse=True)
    if len(result.node_surcharge_m3) > 12:
        node_names = node_names[:12]
    for name in node_names:
        series = result.node_surcharge_m3[name]
        if max(series) > 0.0 or len(result.node_surcharge_m3) <= 12:
            axes[2].plot(time, series, label=name)
            plotted_storage = True
    axes[2].set_ylabel("Node storage\n(m3)")
    axes[2].set_xlabel("Time (min)")
    if plotted_storage:
        axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.25)

    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"Result plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_pyswmm_comparison(
    python_result: SimulationResult,
    comparison: PyswmmComparison,
    save_path: Optional[str] = None,
    max_links: int = 12,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed. Install it with: python3 -m pip install matplotlib")
        return

    link_names = sorted(
        python_result.conduit_flow_m3s,
        key=lambda name: max(
            max(python_result.conduit_flow_m3s.get(name, [0.0])),
            max(comparison.link_flow_m3s.get(name, [0.0]) or [0.0]),
        ),
        reverse=True,
    )[:max_links]
    ncols = 2
    nrows = max(1, ceil(len(link_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), sharex=True, constrained_layout=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, link_name in zip(axes_list, link_names):
        py_series = python_result.conduit_flow_m3s[link_name]
        swmm_series = comparison.link_flow_m3s.get(link_name, [])
        py_time = python_result.time_min
        swmm_time = [i * (py_time[1] - py_time[0]) if len(py_time) > 1 else i for i in range(len(swmm_series))]

        ax.plot(py_time, py_series, color="#1f77b4", linewidth=2.0, label="Python dyn-wave approx")
        ax.plot(swmm_time, swmm_series, color="#d62728", linewidth=1.8, linestyle="--", label="PySWMM")
        ax.set_title(link_name)
        ax.set_ylabel("Flow (m3/s)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes_list[len(link_names):]:
        ax.axis("off")
    for ax in axes_list[-ncols:]:
        ax.set_xlabel("Time (min)")

    fig.suptitle("Pipe flow comparison")
    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"PySWMM comparison plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def print_comparison_diagnostics(
    model: DrainageNetwork,
    comparison: PyswmmComparison,
    top_n: int = 8,
) -> None:
    rows = []
    for conduit_name, row in zip(model.conduits, comparison.summary_rows):
        conduit = model.conduits[conduit_name]
        inlet = model.nodes[conduit.inlet]
        outlet = model.nodes[conduit.outlet]
        slope = (inlet.invert_m - outlet.invert_m) / max(conduit.length_m, 1e-9)
        rows.append(
            (
                abs(row["peak_diff"]),
                conduit_name,
                row,
                slope,
                len(model._in_edges[conduit.inlet]),
                len(model._out_edges[conduit.inlet]),
                conduit.inlet,
                conduit.outlet,
            )
        )

    print("Largest differences diagnostics:")
    print("  link      abs_diff  slope      in/out_edges  inlet -> outlet")
    for abs_diff, conduit_name, row, slope, in_edges, out_edges, inlet, outlet in sorted(rows, reverse=True)[:top_n]:
        print(
            f"  {conduit_name:<8} {abs_diff:>8.4f}  {slope:>9.6f}  "
            f"{in_edges}/{out_edges:<9} {inlet} -> {outlet}"
        )


def run_surface_flood_coupling(
    model: DrainageNetwork,
    result: SimulationResult,
    output_dir: str = "coupled_flood_output",
    dem_path: Optional[str] = None,
    mesh_size_m: float = 60.0,
    max_grid_cells: int = 5000,
    max_frames: int = 12,
    show_interactive_plot: bool = False,
    show_osm_basemap: bool = True,
    show_osm_in_slider: bool = False,
    show_catchments: bool = True,
    show_network: bool = True,
    map_crs: str = "EPSG:2326",
) -> None:
    """Route SWMM node flooding onto the 2D ANUGA surface model from flooding.py."""

    try:
        import flooding
    except (ImportError, SystemExit) as exc:
        print(f"Surface flood coupling skipped: cannot import flooding.py ({exc})")
        return

    runner = getattr(flooding, "run_coupled_surface_flood", None)
    if runner is None:
        print("Surface flood coupling skipped: flooding.py has no run_coupled_surface_flood()")
        return

    try:
        artifacts = runner(
            model=model,
            times_min=result.time_min,
            node_flooding_m3s=result.node_flooding_m3s,
            node_pollutant_conc=result.node_pollutant_conc,
            output_dir=output_dir,
            dem_path=dem_path,
            mesh_size_m=mesh_size_m,
            max_grid_cells=max_grid_cells,
            max_frames=max_frames,
            show_interactive_plot=show_interactive_plot,
            show_osm_basemap=show_osm_basemap,
            show_osm_in_slider=show_osm_in_slider,
            show_catchments=show_catchments,
            show_network=show_network,
            map_crs=map_crs,
        )
    except TypeError:
        # Older SimulationResult has no node flooding time-series field; use the
        # aggregate volumes as a single final pulse rather than silently doing nothing.
        pulse_series = {
            name: [0.0] * max(len(result.time_min), 1)
            for name in model.nodes
        }
        dt_s = 60.0 * (result.time_min[1] - result.time_min[0]) if len(result.time_min) > 1 else 300.0
        for name, volume in result.flooding_m3.items():
            if pulse_series[name]:
                pulse_series[name][-1] = volume / max(dt_s, 1.0)
        artifacts = runner(
            model=model,
            times_min=result.time_min,
            node_flooding_m3s=pulse_series,
            node_pollutant_conc=result.node_pollutant_conc,
            output_dir=output_dir,
            dem_path=dem_path,
            mesh_size_m=mesh_size_m,
            max_grid_cells=max_grid_cells,
            max_frames=max_frames,
            show_interactive_plot=show_interactive_plot,
            show_osm_basemap=show_osm_basemap,
            show_osm_in_slider=show_osm_in_slider,
            show_catchments=show_catchments,
            show_network=show_network,
            map_crs=map_crs,
        )
    except Exception as exc:
        print(f"Surface flood coupling failed: {exc}")
        return

    if artifacts:
        print("Surface flood coupling outputs:")
        for label, path in artifacts.items():
            print(f"  {label}: {path}")


def _float_arg(argv: List[str], name: str, default: float) -> float:
    if name not in argv:
        return default
    idx = argv.index(name)
    if len(argv[idx + 1 : idx + 2]) != 1:
        raise SystemExit(f"Usage: python3 test.py {name} VALUE")
    return float(argv[idx + 1])


def _int_arg(argv: List[str], name: str, default: int) -> int:
    if name not in argv:
        return default
    idx = argv.index(name)
    if len(argv[idx + 1 : idx + 2]) != 1:
        raise SystemExit(f"Usage: python3 test.py {name} VALUE")
    return int(argv[idx + 1])


def _str_arg(argv: List[str], name: str, default: Optional[str]) -> Optional[str]:
    if name not in argv:
        return default
    idx = argv.index(name)
    if len(argv[idx + 1 : idx + 2]) != 1:
        raise SystemExit(f"Usage: python3 test.py {name} VALUE")
    return argv[idx + 1]


def _first_attr(attrs: Dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    lower = {str(k).lower(): v for k, v in attrs.items()}
    for name in names:
        if name in attrs and attrs[name] not in (None, ""):
            return attrs[name]
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nearest_node(x: float, y: float, nodes: Dict[str, Node]) -> str:
    return min(nodes, key=lambda name: (nodes[name].x - x) ** 2 + (nodes[name].y - y) ** 2)


def _read_point_nodes(
    shp_path: str,
    is_outfall: bool,
    name_prefix: str,
) -> List[Node]:
    try:
        import shapefile
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading shapefiles requires pyshp: python3 -m pip install pyshp") from exc

    reader = shapefile.Reader(shp_path)
    fields = [field[0] for field in reader.fields[1:]]
    nodes: List[Node] = []

    for idx, record in enumerate(reader.iterShapeRecords(), start=1):
        attrs = dict(zip(fields, record.record))
        if not record.shape.points:
            continue
        x, y = record.shape.points[0]
        node_id = str(
            _first_attr(attrs, ("MUID", "id", "name", "node", "node_id", "mh_id", "manhole_id"), f"{name_prefix}{idx}")
        )
        invert = _as_float(
            _first_attr(attrs, ("InvertLeve", "InvertLevel", "invert", "invert_m", "inv", "elev", "elevation", "z")),
            0.0,
        )
        ground = _as_float(_first_attr(attrs, ("GroundLeve", "GroundLevel", "ground", "rim", "rim_elev")), invert + 2.0)
        max_depth = _as_float(
            _first_attr(attrs, ("max_depth", "depth", "depth_m", "rim_depth")),
            max(ground - invert, 0.1),
        )
        nodes.append(Node(node_id, invert, max_depth, float(x), float(y), is_outfall=is_outfall))

    return nodes


def build_network_from_shp(
    manhole_shp: str,
    outlet_shp: str,
    pipe_shp: str,
    default_diameter_m: float = 0.6,
    default_roughness_n: float = 0.013,
    endpoint_tolerance_m: float = 5.0,
) -> DrainageNetwork:
    """Build a drainage network from manhole, outlet, and pipe shapefiles.

    Expected useful fields:
    - manhole/outlet: id/name, invert/elevation, max_depth/depth
    - pipe: id/name, from_node, to_node, length, diameter, roughness

    If pipe from/to fields are missing, endpoints are snapped to nearest nodes.
    Shapefile coordinates should be projected in meters, not lon/lat degrees.
    """

    try:
        import shapefile
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading shapefiles requires pyshp: python3 -m pip install pyshp") from exc

    model = DrainageNetwork()
    for node in _read_point_nodes(manhole_shp, is_outfall=False, name_prefix="J"):
        model.add_node(node)
    for node in _read_point_nodes(outlet_shp, is_outfall=True, name_prefix="OUT"):
        model.add_node(node)

    pipe_reader = shapefile.Reader(pipe_shp)
    fields = [field[0] for field in pipe_reader.fields[1:]]

    for idx, record in enumerate(pipe_reader.iterShapeRecords(), start=1):
        attrs = dict(zip(fields, record.record))
        points = record.shape.points
        if len(points) < 2:
            continue

        conduit_id = str(_first_attr(attrs, ("MUID", "id", "name", "pipe", "pipe_id", "link_id"), f"C{idx}"))
        inlet = _first_attr(attrs, ("from_node", "fromnode", "from", "up_node", "upstream"))
        outlet = _first_attr(attrs, ("to_node", "tonode", "to", "down_node", "downstream"))

        if not inlet or not outlet:
            x1, y1 = points[0]
            x2, y2 = points[-1]
            inlet = _nearest_node(float(x1), float(y1), model.nodes)
            outlet = _nearest_node(float(x2), float(y2), model.nodes)
            snap_dist = max(
                sqrt((model.nodes[inlet].x - x1) ** 2 + (model.nodes[inlet].y - y1) ** 2),
                sqrt((model.nodes[outlet].x - x2) ** 2 + (model.nodes[outlet].y - y2) ** 2),
            )
            if snap_dist > endpoint_tolerance_m:
                print(f"Warning: pipe {conduit_id} endpoint snap distance is {snap_dist:.2f} m")

        length_geom = sum(
            sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2)
            for i in range(len(points) - 1)
        )
        length = _as_float(_first_attr(attrs, ("Length", "Length_C", "length", "length_m", "len", "SHAPE_Leng")), length_geom)
        diameter = _as_float(_first_attr(attrs, ("Diameter", "diameter", "diam_m", "diam", "d", "size")), default_diameter_m)
        roughness = _as_float(_first_attr(attrs, ("Manning", "roughness", "n", "manning_n")), default_roughness_n)
        if roughness <= 0.0:
            roughness = default_roughness_n

        vertices = [(float(x), float(y)) for x, y in points]
        inlet = str(inlet)
        outlet = str(outlet)
        if model.nodes[inlet].invert_m < model.nodes[outlet].invert_m:
            inlet, outlet = outlet, inlet
            vertices.reverse()
        model.add_conduit(Conduit(conduit_id, str(inlet), str(outlet), length, diameter, roughness, vertices=vertices))

    return model


def add_fake_subcatchments(
    model: DrainageNetwork,
    area_ha: float = 0.15,
    impervious_fraction: float = 0.55,
    slope: float = 0.012,
) -> None:
    """Attach synthetic subcatchments to every non-outfall node for testing."""

    width_m = sqrt(area_ha * 10000.0)
    for node in model.nodes.values():
        if node.is_outfall:
            continue
        model.add_subcatchment(
            Subcatchment(
                name=f"SC_{node.name}",
                outlet=node.name,
                area_ha=area_ha,
                impervious_fraction=impervious_fraction,
                width_m=width_m,
                slope=slope,
            )
        )


def add_catchments_from_shp(
    model: DrainageNetwork,
    catchment_shp: str,
    impervious_fraction: float = 0.55,
    slope: float = 0.012,
) -> int:
    """Attach polygon catchments to the nearest non-outfall manhole."""

    try:
        import shapefile
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading shapefiles requires pyshp: python3 -m pip install pyshp") from exc

    manholes = {name: node for name, node in model.nodes.items() if not node.is_outfall}
    if not manholes:
        return 0

    reader = shapefile.Reader(catchment_shp)
    fields = [field[0] for field in reader.fields[1:]]
    count = 0

    for idx, record in enumerate(reader.iterShapeRecords(), start=1):
        attrs = dict(zip(fields, record.record))
        points = record.shape.points
        if not points:
            continue
        model.catchment_polygons.append([(float(x), float(y)) for x, y in points])

        cx = _as_float(_first_attr(attrs, ("X_C", "X", "centroid_x", "cx")), sum(p[0] for p in points) / len(points))
        cy = _as_float(_first_attr(attrs, ("Y_C", "Y", "centroid_y", "cy")), sum(p[1] for p in points) / len(points))
        outlet = _nearest_node(cx, cy, manholes)

        area_ha = _as_float(_first_attr(attrs, ("Area_C", "Area_ha", "area_ha")), 0.0)
        if area_ha <= 0.0:
            area_m2 = _as_float(_first_attr(attrs, ("SHAPE_Area", "shape_area", "area_m2")), 0.0)
            area_ha = area_m2 / 10000.0 if area_m2 > 0.0 else 0.1

        catchment_id = str(_first_attr(attrs, ("MUID", "id", "name", "catchment_id"), f"Catchment_{idx}"))
        width_m = sqrt(max(area_ha * 10000.0, 1.0))
        model.add_subcatchment(
            Subcatchment(
                name=catchment_id,
                outlet=outlet,
                area_ha=area_ha,
                impervious_fraction=impervious_fraction,
                width_m=width_m,
                slope=slope,
            )
        )
        count += 1

    return count


def make_outfalls_terminal(model: DrainageNetwork) -> None:
    """Convert non-terminal outlet points into junctions with short terminal outfalls."""

    outlet_nodes = [node for node in list(model.nodes.values()) if node.is_outfall]
    for node in outlet_nodes:
        in_count = len(model._in_edges[node.name])
        out_count = len(model._out_edges[node.name])
        if in_count == 1 and out_count == 0:
            continue

        node.is_outfall = False
        terminal_name = f"{node.name}_OF"
        if terminal_name in model.nodes:
            continue
        terminal = Node(
            terminal_name,
            invert_m=node.invert_m - 0.01,
            max_depth_m=node.max_depth_m,
            x=node.x + 10.0,
            y=node.y + 10.0,
            is_outfall=True,
        )
        model.add_node(terminal)
        model.add_conduit(
            Conduit(
                f"OUT_LINK_{node.name}",
                node.name,
                terminal_name,
                length_m=20.0,
                diameter_m=2.0,
                roughness_n=0.013,
                vertices=[(node.x, node.y), (terminal.x, terminal.y)],
            )
        )


def build_demo_network() -> DrainageNetwork:
    model = DrainageNetwork()

    # Coordinates are schematic layout coordinates, in meters.
    for node in [
        Node("J1", invert_m=12.5, max_depth_m=2.0, x=0, y=120),
        Node("J2", invert_m=11.7, max_depth_m=2.0, x=180, y=160),
        Node("J3", invert_m=10.9, max_depth_m=2.2, x=360, y=110),
        Node("J4", invert_m=10.4, max_depth_m=2.3, x=530, y=150),
        Node("OUT1", invert_m=9.8, max_depth_m=3.0, x=700, y=120, is_outfall=True),
    ]:
        model.add_node(node)

    for conduit in [
        Conduit("C1", "J1", "J2", length_m=190, diameter_m=0.60),
        Conduit("C2", "J2", "J3", length_m=185, diameter_m=0.70),
        Conduit("C3", "J3", "J4", length_m=175, diameter_m=0.80),
        Conduit("C4", "J4", "OUT1", length_m=180, diameter_m=0.90),
    ]:
        model.add_conduit(conduit)

    for subcatchment in [
        Subcatchment("S1", "J1", area_ha=5.0, impervious_fraction=0.55, width_m=110, slope=0.018),
        Subcatchment("S2", "J2", area_ha=4.0, impervious_fraction=0.72, width_m=95, slope=0.014),
        Subcatchment("S3", "J3", area_ha=6.5, impervious_fraction=0.48, width_m=130, slope=0.012),
        Subcatchment("S4", "J4", area_ha=3.5, impervious_fraction=0.65, width_m=90, slope=0.016),
    ]:
        model.add_subcatchment(subcatchment)

    return model


def design_storm(duration_steps: int = 36) -> List[float]:
    """Three-hour storm at 5-minute steps, peak intensity in the middle."""

    base = [0, 4, 8, 15, 22, 35, 52, 74, 96, 80, 58, 40, 28, 20, 14, 8, 4, 2]
    if duration_steps <= len(base):
        return base[:duration_steps]
    return base + [0.0] * (duration_steps - len(base))


def prepare_shp_network(
    manhole_shp: str,
    outlet_shp: str,
    pipe_shp: str,
    use_fake_hydrology: bool,
    catchment_shp: Optional[str] = None,
) -> Tuple[DrainageNetwork, bool]:
    model = build_network_from_shp(manhole_shp, outlet_shp, pipe_shp)
    make_outfalls_terminal(model)

    if catchment_shp and Path(catchment_shp).exists():
        count = add_catchments_from_shp(model, catchment_shp)
        print(f"Catchments loaded: {count} polygons from {catchment_shp}.")
        return model, count > 0

    if use_fake_hydrology:
        add_fake_subcatchments(model)
        print(f"Fake hydrology added: {len(model.subcatchments)} synthetic subcatchments.")
        return model, True
    if not model.subcatchments:
        print("Warning: no subcatchment/inflow data loaded; hydraulic topology is built but event flow will be zero.")
        return model, False
    return model, True


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    run_pyswmm = "--no-compare-pyswmm" not in argv
    approx_only = "--approx-only" in argv
    couple_surface_flood = SURFACE_FLOOD_ENABLED
    if "--couple-surface-flood" in argv:
        couple_surface_flood = True
    if "--no-couple-surface-flood" in argv:
        couple_surface_flood = False
    surface_mesh_size = _float_arg(argv, "--surface-mesh-size", SURFACE_MESH_SIZE_M)
    surface_max_grid_cells = _int_arg(argv, "--surface-max-grid-cells", SURFACE_MAX_GRID_CELLS)
    surface_max_frames = _int_arg(argv, "--surface-max-frames", SURFACE_MAX_FRAMES)
    surface_dem_path = _str_arg(argv, "--surface-dem", SURFACE_DEM_PATH)
    if "--no-surface-dem" in argv:
        surface_dem_path = None
    surface_show_interactive_plot = SURFACE_SHOW_INTERACTIVE_PLOT and "--no-surface-show" not in argv
    surface_show_osm_basemap = SURFACE_SHOW_OSM_BASEMAP and "--no-osm-basemap" not in argv
    surface_show_osm_in_slider = (
        (SURFACE_SHOW_OSM_IN_SLIDER or "--osm-slider" in argv)
        and "--no-osm-slider" not in argv
    )
    surface_show_catchments = SURFACE_SHOW_CATCHMENTS and "--no-surface-catchments" not in argv
    surface_show_network = SURFACE_SHOW_NETWORK and "--no-surface-network" not in argv
    surface_map_crs = _str_arg(argv, "--surface-map-crs", SURFACE_MAP_CRS)
    run_pyswmm = run_pyswmm and not approx_only
    use_fake_hydrology = "--no-fake-hydrology" not in argv
    has_hydrologic_input = True

    default_shps = ("Manhole100.shp", "outlet100.shp", "pipe100.shp")
    auto_shp = all(Path(path).exists() for path in default_shps)
    catchment_shp = None
    if "--catchment" in argv:
        cidx = argv.index("--catchment")
        if len(argv[cidx + 1 : cidx + 2]) != 1:
            raise SystemExit("Usage: python3 test.py --catchment catchment.shp")
        catchment_shp = argv[cidx + 1]
    elif Path("catchment.shp").exists():
        catchment_shp = "catchment.shp"

    if "--from-shp" in argv:
        idx = argv.index("--from-shp")
        try:
            manhole_shp, outlet_shp, pipe_shp = argv[idx + 1 : idx + 4]
        except ValueError:
            raise SystemExit("Usage: python3 test.py --from-shp manhole.shp outlet.shp pipe.shp")
        if len(argv[idx + 1 : idx + 4]) != 3:
            raise SystemExit("Usage: python3 test.py --from-shp manhole.shp outlet.shp pipe.shp")
        model, has_hydrologic_input = prepare_shp_network(
            manhole_shp,
            outlet_shp,
            pipe_shp,
            use_fake_hydrology,
            catchment_shp=catchment_shp,
        )
    elif auto_shp and "--demo" not in argv:
        print("Auto-detected shapefiles: Manhole100.shp, outlet100.shp, pipe100.shp")
        model, has_hydrologic_input = prepare_shp_network(*default_shps, use_fake_hydrology, catchment_shp=catchment_shp)
    else:
        model = build_demo_network()

    rainfall = design_storm()
    report_dt_s = 300.0
    approx_result = model.simulate_dynamic_wave(rainfall, report_dt_s=report_dt_s, routing_dt_s=30.0)
    result = approx_result
    node_runoff = local_runoff_by_node(model, rainfall, dt_s=report_dt_s)
    model.write_swmm_inp(
        "demo_network.inp",
        node_runoff_m3s=node_runoff,
        rainfall_mm_h=rainfall,
        dt_s=report_dt_s,
        use_native_hydrology=SWMM_NATIVE_WATER_QUALITY_ENABLED,
        pollutants=SWMM_WQ_POLLUTANTS if SWMM_NATIVE_WATER_QUALITY_ENABLED else None,
    )

    print("Python dynamic-wave approximation finished")
    print(f"Nodes: {len(model.nodes)}, conduits: {len(model.conduits)}, subcatchments: {len(model.subcatchments)}")
    print(f"Approx continuity error: {approx_result.continuity_error_pct:.4f}%")
    print("Approx flooding volume by node (m3):")
    flooded_nodes = sorted(
        ((node, volume) for node, volume in approx_result.flooding_m3.items() if volume > 1e-6),
        key=lambda item: item[1],
        reverse=True,
    )
    for node, volume in flooded_nodes[:20]:
        print(f"  {node}: {volume:.2f}")
    if len(flooded_nodes) > 20:
        print(f"  ... {len(flooded_nodes) - 20} more flooded nodes omitted")
    if not flooded_nodes:
        print("  none")
    print("SWMM-compatible input exported to demo_network.inp")

    if run_pyswmm and has_hydrologic_input:
        comparison = compare_with_pyswmm(model, approx_result, dt_s=report_dt_s, inp_path="demo_network.inp")
        if comparison is not None:
            result = approx_result._replace(
                conduit_flow_m3s=comparison.link_flow_m3s,
                node_flooding_m3s=comparison.node_flooding_m3s,
                node_pollutant_conc=comparison.node_pollutant_conc,
                flooding_m3={
                    name: sum(series) * report_dt_s
                    for name, series in comparison.node_flooding_m3s.items()
                },
                continuity_error_pct=0.0,
            )
            print("Primary result: official SWMM DYNWAVE via PySWMM.")
            swmm_flooded_nodes = sorted(
                ((node, volume) for node, volume in result.flooding_m3.items() if volume > 1e-6),
                key=lambda item: item[1],
                reverse=True,
            )
            print("PySWMM flooding volume by node (m3):")
            for node, volume in swmm_flooded_nodes[:20]:
                print(f"  {node}: {volume:.2f}")
            if len(swmm_flooded_nodes) > 20:
                print(f"  ... {len(swmm_flooded_nodes) - 20} more flooded nodes omitted")
            if not swmm_flooded_nodes:
                print("  none")
            ranked_rows = sorted(
                zip(model.conduits, comparison.summary_rows),
                key=lambda item: max(item[1]["python_peak"], item[1]["pyswmm_peak"]),
                reverse=True,
            )
            rows_to_print = ranked_rows[:12]
            print("Pipe-flow comparison, Python dyn-wave approx vs PySWMM (top pipes by peak flow, m3/s):")
            print("  link      py_peak   swmm_peak  diff      py_mean   swmm_mean diff")
            for conduit_name, row in rows_to_print:
                print(
                    f"  {conduit_name:<8} "
                    f"{row['python_peak']:>7.4f}  {row['pyswmm_peak']:>9.4f}  {row['peak_diff']:>8.4f}  "
                    f"{row['python_mean']:>7.4f}  {row['pyswmm_mean']:>9.4f}  {row['mean_diff']:>8.4f}"
                )
            if len(ranked_rows) > len(rows_to_print):
                print(f"  ... {len(ranked_rows) - len(rows_to_print)} more pipes omitted from console table")
            print_comparison_diagnostics(model, comparison)
            plot_pyswmm_comparison(approx_result, comparison, save_path="pyswmm_comparison.png")
    else:
        reason = "--approx-only/--no-compare-pyswmm" if not run_pyswmm else "missing subcatchment/inflow data"
        print(f"PySWMM comparison skipped ({reason}).")

    if couple_surface_flood:
        run_surface_flood_coupling(
            model,
            result,
            output_dir=SURFACE_FLOOD_OUTPUT_DIR,
            dem_path=surface_dem_path,
            mesh_size_m=surface_mesh_size,
            max_grid_cells=surface_max_grid_cells,
            max_frames=surface_max_frames,
            show_interactive_plot=surface_show_interactive_plot,
            show_osm_basemap=surface_show_osm_basemap,
            show_osm_in_slider=surface_show_osm_in_slider,
            show_catchments=surface_show_catchments,
            show_network=surface_show_network,
            map_crs=surface_map_crs,
        )

    plot_network(model, result, save_path="network_plot.png")
    plot_results(result, save_path="simulation_results.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pyswmm-worker":
        raise SystemExit(pyswmm_worker(sys.argv))
    main()
