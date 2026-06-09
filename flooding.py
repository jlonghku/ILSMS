"""
ANUGA flood-depth demo.

Run this file in the Python environment where ANUGA is installed:

    python3 test.py

Outputs are written to ``anuga_demo_output/``:

* simple_flood.sww                  ANUGA time-series result
* final_flood_depth.png             final water-depth map
* flood_depth_timeseries.html       interactive 2D depth viewer with slider
* final_tracer_concentration.png    final conservative tracer map
* tracer_timeseries.html            interactive conservative tracer viewer
* final_nh4_n_concentration.png     final ammonia nitrogen map
* nh4_n_timeseries.html             interactive ammonia nitrogen viewer

The script also opens a Matplotlib window with a time slider by default. Use
``--no-show`` to generate files only.
"""

from __future__ import annotations

import argparse
import base64
import math
from io import BytesIO
from pathlib import Path
import urllib.request

try:
    import matplotlib
    import numpy as np
    from netCDF4 import Dataset
except ImportError as exc:
    raise SystemExit(
        "Missing Python dependency: "
        f"{exc.name}\n\n"
        "Run this script in the environment where ANUGA is installed, or install:\n"
        "    pip install anuga matplotlib netCDF4 numpy"
    ) from exc

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.widgets import RadioButtons, Slider


OUTDIR = Path("anuga_demo_output")
SIM_NAME = "simple_flood"
SWW_FILE = OUTDIR / f"{SIM_NAME}.sww"
FINAL_PNG = OUTDIR / "final_flood_depth.png"
HTML_VIEWER = OUTDIR / "flood_depth_timeseries.html"

MIN_WET_DEPTH = 0.02

WATER_QUALITY_SPECIES = [
    {
        "key": "tracer",
        "menu_label": "Tracer",
        "title": "Conservative tracer concentration",
        "label": "Concentration (relative units)",
        "cmap": "inferno",
        "source_x": 180.0,
        "source_y": 150.0,
        "source_radius": 35.0,
        "release_duration": 260.0,
        "source_rate": 0.035,
        "diffusivity": 4.0,
        "initial_background": 0.0,
        "equation": "conservative",
    },
    {
        "key": "nh4_n",
        "menu_label": "NH4-N",
        "title": "Ammonia nitrogen (NH4-N)",
        "label": "NH4-N (mg/L, demo units)",
        "cmap": "viridis",
        "source_x": 180.0,
        "source_y": 150.0,
        "source_radius": 35.0,
        "release_duration": 420.0,
        "source_rate": 0.025,
        "diffusivity": 3.0,
        "initial_background": 0.03,
        "equation": "nh4_simple",
        "nitrification_rate": 7.0e-4,
        "settling_loss_rate": 1.0e-4,
        "background_concentration": 0.03,
        "background_exchange_rate": 5.0e-5,
    },
]


def conservative_equation(concentration, dt, species):
    """No local reaction; transport and diffusion only."""
    return concentration


def nh4_simple_equation(concentration, dt, species):
    """
    Simple ammonia nitrogen equation after advection and diffusion.

    dC/dt = -(k_nitrification + k_settle) C
            + k_background (C_background - C)
    """
    nitrification_rate = float(species.get("nitrification_rate", 0.0))
    settling_loss_rate = float(species.get("settling_loss_rate", 0.0))
    background_exchange_rate = float(species.get("background_exchange_rate", 0.0))
    background_concentration = float(species.get("background_concentration", 0.0))

    loss_rate = nitrification_rate + settling_loss_rate
    if loss_rate > 0.0:
        concentration *= np.exp(-loss_rate * dt)

    if background_exchange_rate > 0.0:
        relaxation = 1.0 - np.exp(-background_exchange_rate * dt)
        concentration += relaxation * (background_concentration - concentration)

    return np.maximum(concentration, 0.0)


REACTION_EQUATIONS = {
    "conservative": conservative_equation,
    "nh4_simple": nh4_simple_equation,
}


def elevation(x, y):
    """Downstream-sloping bed with one central mound."""
    bed = 5.0 - 0.003 * x
    bump = 0.8 * np.exp(-((x - 500.0) ** 2 + (y - 150.0) ** 2) / 8000.0)
    return bed + bump


def _model_bounds(model, padding):
    xs = [node.x for node in model.nodes.values()]
    ys = [node.y for node in model.nodes.values()]
    for polygon in getattr(model, "catchment_polygons", []):
        xs.extend(point[0] for point in polygon)
        ys.extend(point[1] for point in polygon)

    if not xs or not ys:
        return 0.0, 1000.0, 0.0, 300.0

    min_x = float(min(xs) - padding)
    max_x = float(max(xs) + padding)
    min_y = float(min(ys) - padding)
    max_y = float(max(ys) + padding)
    if max_x <= min_x:
        max_x = min_x + 100.0
    if max_y <= min_y:
        max_y = min_y + 100.0
    return min_x, max_x, min_y, max_y


def _network_surface_elevation(model, x_origin, y_origin):
    """Create a smooth ground surface from drainage-node rim elevations."""

    nodes = list(model.nodes.values())
    node_x = np.asarray([node.x - x_origin for node in nodes], dtype=float)
    node_y = np.asarray([node.y - y_origin for node in nodes], dtype=float)
    node_z = np.asarray([node.invert_m + node.max_depth_m for node in nodes], dtype=float)

    def surface(x, y):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        values = np.zeros_like(x_arr, dtype=float)
        weights = np.zeros_like(x_arr, dtype=float)
        for nx, ny, nz in zip(node_x, node_y, node_z):
            dist2 = (x_arr - nx) ** 2 + (y_arr - ny) ** 2
            weight = 1.0 / np.maximum(dist2, 25.0)
            values += weight * nz
            weights += weight
        return values / np.maximum(weights, 1.0e-12)

    return surface


def read_ascii_dem(dem_path):
    """Read an ESRI ASCII grid DEM."""

    dem_path = Path(dem_path)
    header = {}
    with dem_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(6):
            key, value = handle.readline().split()[:2]
            header[key.lower()] = float(value)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    values = np.loadtxt(dem_path, skiprows=6, dtype=float)
    if values.shape != (nrows, ncols):
        values = values.reshape((nrows, ncols))

    nodata = float(header.get("nodata_value", -9999.0))
    values = np.where(values == nodata, np.nan, values)
    return {
        "path": str(dem_path),
        "ncols": ncols,
        "nrows": nrows,
        "xllcorner": float(header["xllcorner"]),
        "yllcorner": float(header["yllcorner"]),
        "cellsize": float(header["cellsize"]),
        "nodata": nodata,
        "values": values,
    }


def sample_ascii_dem(dem, x_abs, y_abs):
    """Bilinearly sample ESRI ASCII grid values at absolute coordinates."""

    x = np.asarray(x_abs, dtype=float)
    y = np.asarray(y_abs, dtype=float)
    cell = dem["cellsize"]
    col = (x - dem["xllcorner"]) / cell - 0.5
    row = (dem["yllcorner"] + dem["nrows"] * cell - y) / cell - 0.5

    col0 = np.floor(col).astype(int)
    row0 = np.floor(row).astype(int)
    col1 = col0 + 1
    row1 = row0 + 1
    valid = (
        (col0 >= 0)
        & (row0 >= 0)
        & (col1 < dem["ncols"])
        & (row1 < dem["nrows"])
    )

    out = np.full(x.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out

    values = dem["values"]
    c0 = col0[valid]
    c1 = col1[valid]
    r0 = row0[valid]
    r1 = row1[valid]
    wx = col[valid] - c0
    wy = row[valid] - r0

    z00 = values[r0, c0]
    z10 = values[r0, c1]
    z01 = values[r1, c0]
    z11 = values[r1, c1]
    sample_valid = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
    sampled = (
        z00 * (1.0 - wx) * (1.0 - wy)
        + z10 * wx * (1.0 - wy)
        + z01 * (1.0 - wx) * wy
        + z11 * wx * wy
    )

    valid_indices = np.flatnonzero(valid)
    out[valid_indices[sample_valid]] = sampled[sample_valid]
    return out


def _dem_surface_elevation(model, x_origin, y_origin, dem_path):
    """Create a surface from DEM values, falling back to node-rim interpolation."""

    dem = read_ascii_dem(dem_path)
    fallback_surface = _network_surface_elevation(model, x_origin, y_origin)

    def surface(x, y):
        fallback = np.asarray(fallback_surface(x, y), dtype=float)
        dem_values = sample_ascii_dem(dem, np.asarray(x, dtype=float) + x_origin, np.asarray(y, dtype=float) + y_origin)
        return np.where(np.isfinite(dem_values), dem_values, fallback)

    return surface, dem


def _domain_centroid_coordinates(domain):
    if hasattr(domain, "get_centroid_coordinates"):
        coords = domain.get_centroid_coordinates()
        return np.asarray(coords[:, 0], dtype=float), np.asarray(coords[:, 1], dtype=float)
    coords = np.asarray(domain.centroid_coordinates, dtype=float)
    return coords[:, 0], coords[:, 1]


def _add_stage_volume(domain, mask, volume_m3):
    if volume_m3 <= 0.0 or not np.any(mask):
        return
    stage = domain.quantities["stage"].centroid_values
    elevation_values = domain.quantities["elevation"].centroid_values
    area = domain.areas
    depth_increment = volume_m3 / max(float(np.sum(area[mask])), 1.0e-9)
    stage[mask] = np.maximum(stage[mask], elevation_values[mask]) + depth_increment


_OSM_TILE_CACHE = {}
_OSM_WARNING_PRINTED = False
_BASEMAP_TILE_URL = "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
_BASEMAP_USER_AGENT = "pyswmm-anuga-flood-visualizer/1.0"


def _lonlat_to_tile(lon, lat, zoom):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi)
        / 2.0
        * n
    )
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _tile_bounds_lonlat(x, y, zoom):
    n = 2**zoom
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


def _warn_osm_once(message):
    global _OSM_WARNING_PRINTED
    if not _OSM_WARNING_PRINTED:
        print(f"OSM basemap skipped: {message}")
        _OSM_WARNING_PRINTED = True


def _draw_osm_basemap(ax, x_origin, y_origin, map_crs, zoom=14):
    try:
        from pyproj import Transformer
        import matplotlib.image as mpimg
    except Exception as exc:
        _warn_osm_once(f"pyproj/matplotlib image support unavailable ({exc})")
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    to_lonlat = Transformer.from_crs(map_crs, "EPSG:4326", always_xy=True)
    from_lonlat = Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)

    corners_x = [xmin + x_origin, xmax + x_origin, xmin + x_origin, xmax + x_origin]
    corners_y = [ymin + y_origin, ymin + y_origin, ymax + y_origin, ymax + y_origin]
    lons, lats = to_lonlat.transform(corners_x, corners_y)
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    if not all(np.isfinite([west, east, south, north])):
        _warn_osm_once("map coordinates could not be transformed to lon/lat")
        return

    x0, y0 = _lonlat_to_tile(west, north, zoom)
    x1, y1 = _lonlat_to_tile(east, south, zoom)
    tile_count = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
    if tile_count > 16 and zoom > 12:
        return _draw_osm_basemap(ax, x_origin, y_origin, map_crs, zoom=zoom - 1)
    if tile_count > 30:
        _warn_osm_once(f"tile count too large at zoom {zoom} ({tile_count})")
        return

    for tx in range(min(x0, x1), max(x0, x1) + 1):
        for ty in range(min(y0, y1), max(y0, y1) + 1):
            cache_key = (zoom, tx, ty)
            if cache_key not in _OSM_TILE_CACHE:
                url = _BASEMAP_TILE_URL.format(z=zoom, x=tx, y=ty)
                try:
                    request = urllib.request.Request(
                        url,
                        headers={"User-Agent": _BASEMAP_USER_AGENT},
                    )
                    with urllib.request.urlopen(request, timeout=6) as response:
                        image_bytes = BytesIO(response.read())
                    _OSM_TILE_CACHE[cache_key] = mpimg.imread(image_bytes, format="png")
                except Exception as exc:
                    _warn_osm_once(str(exc))
                    return

            lon_w, lat_s, lon_e, lat_n = _tile_bounds_lonlat(tx, ty, zoom)
            xs, ys = from_lonlat.transform([lon_w, lon_e], [lat_s, lat_n])
            ax.imshow(
                _OSM_TILE_CACHE[cache_key],
                extent=(xs[0] - x_origin, xs[1] - x_origin, ys[0] - y_origin, ys[1] - y_origin),
                origin="upper",
                zorder=-20,
                alpha=0.95,
            )


def _draw_catchment_basemap(ax, model, x_origin, y_origin):
    if model is None or not getattr(model, "catchment_polygons", None):
        return
    try:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon
    except Exception:
        return

    patches = []
    for polygon in model.catchment_polygons:
        if len(polygon) < 3:
            continue
        local_points = [(x - x_origin, y - y_origin) for x, y in polygon]
        patches.append(Polygon(local_points, closed=True))
    if not patches:
        return
    collection = PatchCollection(
        patches,
        facecolor="#f3d35b",
        edgecolor="#7f7448",
        linewidth=0.35,
        alpha=0.22,
        zorder=-5,
    )
    ax.add_collection(collection)


def _draw_network_overlay(ax, model, x_origin, y_origin):
    if model is None:
        return
    for conduit in model.conduits.values():
        n1 = model.nodes[conduit.inlet]
        n2 = model.nodes[conduit.outlet]
        vertices = conduit.vertices or [(n1.x, n1.y), (n2.x, n2.y)]
        xs = [x - x_origin for x, _y in vertices]
        ys = [y - y_origin for _x, y in vertices]
        ax.plot(xs, ys, color="#1f5f8b", linewidth=0.75, alpha=0.80, zorder=9)
    for node in model.nodes.values():
        marker = "s" if node.is_outfall else "o"
        color = "#1b9e77" if node.is_outfall else "#d95f02"
        ax.scatter(
            node.x - x_origin,
            node.y - y_origin,
            s=12 if len(model.nodes) > 80 else 24,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.90,
            zorder=10,
        )


def build_depth_viewer_from_sww(
    sww_file,
    output_dir,
    prefix="surface_flood",
    max_frames=36,
    show_interactive_plot=False,
    water_quality_species=None,
    overlay_model=None,
    x_origin=0.0,
    y_origin=0.0,
    show_osm_basemap=True,
    show_osm_in_slider=False,
    show_catchments=True,
    show_network=True,
    map_crs="EPSG:2326",
):
    """Create depth and water-quality outputs for a coupled SWW file."""

    output_dir = Path(output_dir)
    x, y, triangles, times, depths, xmomentum, ymomentum = read_sww_results(Path(sww_file))
    triangulation = mtri.Triangulation(x, y, triangles)
    depth_vmax = depth_color_limit(depths)

    final_png = output_dir / f"final_{prefix}_depth.png"
    html_viewer = output_dir / f"{prefix}_depth_timeseries.html"

    final_frame = render_scalar_frame(
        triangulation,
        depths[-1],
        times[-1],
        depth_vmax,
        "PySWMM-coupled flood depth",
        "Water depth (m)",
        "Blues",
        show_mesh=True,
        overlay_model=overlay_model,
        x_origin=x_origin,
        y_origin=y_origin,
        show_osm_basemap=show_osm_basemap,
        show_catchments=show_catchments,
        show_network=show_network,
        map_crs=map_crs,
    )
    final_png.write_bytes(base64.b64decode(final_frame))

    if max_frames and len(times) > max_frames:
        frame_indices = np.linspace(0, len(times) - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(len(times), dtype=int)

    frame_images = [
        render_scalar_frame(
            triangulation,
            depths[index],
            times[index],
            depth_vmax,
            "PySWMM-coupled flood depth",
            "Water depth (m)",
            "Blues",
            overlay_model=overlay_model,
            x_origin=x_origin,
            y_origin=y_origin,
            show_osm_basemap=show_osm_in_slider,
            show_catchments=show_catchments,
            show_network=show_network,
            map_crs=map_crs,
        )
        for index in frame_indices
    ]
    write_interactive_html(
        times[frame_indices],
        frame_images,
        html_viewer,
        "PySWMM-coupled flood depth through time",
        "2D flood depth map from PySWMM node flooding",
    )

    artifacts = {"final_png": str(final_png), "html_viewer": str(html_viewer)}
    plot_datasets = [
        {
            "menu_label": "Flood depth",
            "title": "PySWMM-coupled flood depth",
            "label": "Water depth (m)",
            "cmap": "Blues",
            "values": depths,
            "vmax": depth_vmax,
        }
    ]

    species_configs = water_quality_species if water_quality_species is not None else WATER_QUALITY_SPECIES
    water_quality = simulate_water_quality(
        times,
        depths,
        xmomentum,
        ymomentum,
        triangulation,
        species_configs=species_configs,
    )

    for species in species_configs:
        values = water_quality[species["key"]]
        species_vmax = scalar_color_limit(values)
        species_final_png = output_dir / f"final_{prefix}_{species['key']}_concentration.png"
        species_html_viewer = output_dir / f"{prefix}_{species['key']}_timeseries.html"

        species_final_frame = render_scalar_frame(
            triangulation,
            values[-1],
            times[-1],
            species_vmax,
            species["title"],
            species["label"],
            species["cmap"],
            show_mesh=True,
            overlay_model=overlay_model,
            x_origin=x_origin,
            y_origin=y_origin,
            show_osm_basemap=show_osm_basemap,
            show_catchments=show_catchments,
            show_network=show_network,
            map_crs=map_crs,
        )
        species_final_png.write_bytes(base64.b64decode(species_final_frame))

        species_frame_images = [
            render_scalar_frame(
                triangulation,
                values[index],
                times[index],
                species_vmax,
                species["title"],
                species["label"],
                species["cmap"],
                overlay_model=overlay_model,
                x_origin=x_origin,
                y_origin=y_origin,
                show_osm_basemap=show_osm_basemap,
                show_catchments=show_catchments,
                show_network=show_network,
                map_crs=map_crs,
            )
            for index in frame_indices
        ]
        write_interactive_html(
            times[frame_indices],
            species_frame_images,
            species_html_viewer,
            f"{species['title']} through time",
            f"2D {species['title']} map",
        )

        artifacts[f"{species['key']}_final_png"] = str(species_final_png)
        artifacts[f"{species['key']}_html_viewer"] = str(species_html_viewer)
        plot_datasets.append(
            {
                "menu_label": species.get("menu_label", species["key"]),
                "title": species["title"],
                "label": species["label"],
                "cmap": species["cmap"],
                "values": values,
                "vmax": species_vmax,
            }
        )

    if show_interactive_plot:
        show_combined_slider_plot(
            triangulation,
            times,
            plot_datasets,
            overlay_model=overlay_model,
            x_origin=x_origin,
            y_origin=y_origin,
            show_osm_basemap=show_osm_basemap,
            show_catchments=show_catchments,
            show_network=show_network,
            map_crs=map_crs,
        )
        plt.show()
    return artifacts


def _species_template_for_pollutant(pollutant_name):
    name = str(pollutant_name)
    normalized = name.lower().replace("-", "_")
    if normalized in ("nh4n", "nh4_n", "ammonia", "ammonia_n"):
        template = next((dict(species) for species in WATER_QUALITY_SPECIES if species["key"] == "nh4_n"), {})
        template.update(
            {
                "key": "nh4_n",
                "menu_label": "NH4-N",
                "title": "PySWMM NH4-N",
                "label": "NH4-N (mg/L)",
                "equation": "nh4_simple",
            }
        )
        return template

    template = next((dict(species) for species in WATER_QUALITY_SPECIES if species["key"] == "tracer"), {})
    template.update(
        {
            "key": normalized,
            "menu_label": name,
            "title": f"PySWMM {name}",
            "label": f"{name} (mg/L)",
            "equation": "conservative",
        }
    )
    return template


def _build_pyswmm_water_quality_species(
    model,
    active_nodes,
    node_flooding_m3s,
    node_pollutant_conc,
    x_origin,
    y_origin,
    finaltime_s,
    yieldstep_s,
):
    pollutant_names = sorted(
        {
            pollutant_name
            for node_data in node_pollutant_conc.values()
            for pollutant_name, series in node_data.items()
            if series and max(series) > 0.0
        }
    )
    if not pollutant_names:
        return []

    species_configs = []
    for pollutant_name in pollutant_names:
        species = _species_template_for_pollutant(pollutant_name)
        species["source_rate"] = 0.0
        species["release_duration"] = finaltime_s
        species["surface_sources"] = []
        for node_name in active_nodes:
            node = model.nodes[node_name]
            flow_series = list(node_flooding_m3s.get(node_name, []))
            concentration_series = list(
                node_pollutant_conc.get(node_name, {}).get(pollutant_name, [])
            )
            if not flow_series or not concentration_series:
                continue
            if max(flow_series) <= 0.0 or max(concentration_series) <= 0.0:
                continue
            species["surface_sources"].append(
                {
                    "x": node.x - x_origin,
                    "y": node.y - y_origin,
                    "radius": float(species.get("source_radius", 35.0)),
                    "flow_m3s": flow_series,
                    "concentration": concentration_series,
                }
            )

        if species["surface_sources"]:
            total_mass_proxy = 0.0
            for source in species["surface_sources"]:
                total_mass_proxy += sum(
                    q * c * yieldstep_s
                    for q, c in zip(source["flow_m3s"], source["concentration"])
                )
            print(
                f"PySWMM pollutant coupled to surface: {pollutant_name}, "
                f"sources={len(species['surface_sources'])}, mass_proxy={total_mass_proxy:.3f}"
            )
            species_configs.append(species)

    return species_configs


def run_coupled_surface_flood(
    model,
    times_min,
    node_flooding_m3s,
    node_pollutant_conc=None,
    output_dir="coupled_flood_output",
    sim_name="pyswmm_coupled_flood",
    dem_path=None,
    mesh_size_m=60.0,
    max_grid_cells=5000,
    source_radius_m=18.0,
    max_frames=36,
    show_interactive_plot=False,
    show_osm_basemap=True,
    show_osm_in_slider=False,
    show_catchments=True,
    show_network=True,
    map_crs="EPSG:2326",
):
    """Run ANUGA using PySWMM node flooding rates as surface point sources.

    ``node_flooding_m3s`` is expected to be a dict of node-name -> m3/s series,
    aligned with ``times_min`` from ``test.py``.
    """

    try:
        import anuga
    except ImportError as exc:
        raise RuntimeError(
            "ANUGA is not installed. Install/run in an ANUGA environment to use "
            "--couple-surface-flood."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    node_pollutant_conc = node_pollutant_conc or {}

    times_min = list(times_min)
    if not times_min:
        raise ValueError("times_min cannot be empty")
    if len(times_min) > 1:
        yieldstep_s = float((times_min[1] - times_min[0]) * 60.0)
    else:
        yieldstep_s = 300.0
    finaltime_s = float(times_min[-1] * 60.0 + yieldstep_s)

    active_nodes = [
        node_name
        for node_name, series in node_flooding_m3s.items()
        if node_name in model.nodes and max(series or [0.0]) > 0.0
    ]
    if not active_nodes:
        note = output_dir / "no_surface_flooding.txt"
        note.write_text("No positive PySWMM node flooding was available for 2D routing.\n", encoding="utf-8")
        return {"note": str(note)}

    min_x, max_x, min_y, max_y = _model_bounds(model, padding=4.0 * mesh_size_m)
    width = max_y - min_y
    length = max_x - min_x
    nx = max(2, int(np.ceil(length / mesh_size_m)))
    ny = max(2, int(np.ceil(width / mesh_size_m)))
    max_grid_cells = max(4, int(max_grid_cells or 0))
    grid_cells = nx * ny
    if grid_cells > max_grid_cells:
        scale = np.sqrt(grid_cells / max_grid_cells)
        mesh_size_m *= float(scale)
        nx = max(2, int(np.ceil(length / mesh_size_m)))
        ny = max(2, int(np.ceil(width / mesh_size_m)))
        print(
            "Surface mesh was coarsened to respect max_grid_cells: "
            f"mesh_size={mesh_size_m:.2f} m, nx={nx}, ny={ny}, cells={nx * ny}"
        )
    else:
        print(f"Surface mesh: mesh_size={mesh_size_m:.2f} m, nx={nx}, ny={ny}, cells={grid_cells}")

    largest_source_node = max(
        active_nodes,
        key=lambda name: sum(node_flooding_m3s.get(name, [])) * yieldstep_s,
    )
    coupled_species = _build_pyswmm_water_quality_species(
        model,
        active_nodes,
        node_flooding_m3s,
        node_pollutant_conc,
        min_x,
        min_y,
        finaltime_s,
        yieldstep_s,
    )
    print(
        "Largest PySWMM flooding node for water-quality coupling: "
        f"{largest_source_node} ({sum(node_flooding_m3s.get(largest_source_node, [])) * yieldstep_s:.2f} m3)"
    )

    points, vertices, boundary = anuga.rectangular_cross(nx, ny, len1=length, len2=width)
    domain = anuga.Domain(points, vertices, boundary)
    domain.set_name(sim_name)
    domain.set_datadir(str(output_dir))
    domain.set_flow_algorithm("DE0")

    if dem_path and Path(dem_path).exists():
        surface, dem = _dem_surface_elevation(model, min_x, min_y, dem_path)
        sample_x = np.asarray([node.x for node in model.nodes.values()], dtype=float)
        sample_y = np.asarray([node.y for node in model.nodes.values()], dtype=float)
        dem_at_nodes = sample_ascii_dem(dem, sample_x, sample_y)
        valid_fraction = float(np.mean(np.isfinite(dem_at_nodes))) if len(dem_at_nodes) else 0.0
        if valid_fraction <= 0.0:
            print(
                "DEM has no valid overlap with model node coordinates; "
                "using node-rim interpolated surface instead."
            )
            surface = _network_surface_elevation(model, min_x, min_y)
        else:
            print(f"DEM valid at {100.0 * valid_fraction:.1f}% of model nodes.")
            print(
                "Surface elevation loaded from DEM: "
                f"{dem['path']} ({dem['ncols']} x {dem['nrows']}, cellsize={dem['cellsize']})"
            )
    elif dem_path:
        print(f"DEM not found, using node-rim interpolated surface: {dem_path}")
        surface = _network_surface_elevation(model, min_x, min_y)
    else:
        print("No DEM configured; using node-rim interpolated surface.")
        surface = _network_surface_elevation(model, min_x, min_y)
    domain.set_quantity("elevation", surface)
    domain.set_quantity("friction", 0.035)
    domain.set_quantity("stage", expression="elevation")
    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary({tag: reflective for tag in ("left", "right", "top", "bottom")})

    cx, cy = _domain_centroid_coordinates(domain)
    source_masks = {}
    for node_name in active_nodes:
        node = model.nodes[node_name]
        local_x = node.x - min_x
        local_y = node.y - min_y
        radius2 = source_radius_m * source_radius_m
        mask = (cx - local_x) ** 2 + (cy - local_y) ** 2 <= radius2
        if not np.any(mask):
            nearest = int(np.argmin((cx - local_x) ** 2 + (cy - local_y) ** 2))
            mask = np.zeros_like(cx, dtype=bool)
            mask[nearest] = True
        source_masks[node_name] = mask

    for domain_time in domain.evolve(yieldstep=yieldstep_s, finaltime=finaltime_s):
        step = min(int(domain_time // yieldstep_s), len(times_min) - 1)
        for node_name, mask in source_masks.items():
            series = node_flooding_m3s.get(node_name, [])
            rate = float(series[step]) if step < len(series) else 0.0
            _add_stage_volume(domain, mask, rate * yieldstep_s)
        print(domain.timestepping_statistics())

    sww_file = output_dir / f"{sim_name}.sww"
    artifacts = {"sww_file": str(sww_file)}
    artifacts.update(
        build_depth_viewer_from_sww(
            sww_file,
            output_dir,
            prefix=sim_name,
            max_frames=max_frames,
            show_interactive_plot=show_interactive_plot,
            water_quality_species=coupled_species,
            overlay_model=model,
            x_origin=min_x,
            y_origin=min_y,
            show_osm_basemap=show_osm_basemap,
            show_osm_in_slider=show_osm_in_slider,
            show_catchments=show_catchments,
            show_network=show_network,
            map_crs=map_crs,
        )
    )
    return artifacts


def run_simulation():
    """Create and run a small rectangular ANUGA flood model."""
    try:
        import anuga
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependency: anuga\n\n"
            "Run this script in the environment where ANUGA is installed, or install:\n"
            "    pip install anuga"
        ) from exc

    OUTDIR.mkdir(exist_ok=True)

    length = 1000.0
    width = 300.0
    dx = dy = 20.0

    points, vertices, boundary = anuga.rectangular_cross(
        int(length / dx),
        int(width / dy),
        len1=length,
        len2=width,
    )

    domain = anuga.Domain(points, vertices, boundary)
    domain.set_name(SIM_NAME)
    domain.set_datadir(str(OUTDIR))
    domain.set_flow_algorithm("DE0")

    domain.set_quantity("elevation", elevation)
    domain.set_quantity("friction", 0.035)
    domain.set_quantity("stage", expression="elevation")

    reflective = anuga.Reflective_boundary(domain)
    downstream = anuga.Dirichlet_boundary([2.2, 0.0, 0.0])
    upstream = anuga.Dirichlet_boundary([5.3, 2.0, 0.0])

    domain.set_boundary(
        {
            "left": upstream,
            "right": downstream,
            "top": reflective,
            "bottom": reflective,
        }
    )

    for _ in domain.evolve(yieldstep=20.0, finaltime=1200.0):
        print(domain.timestepping_statistics())


def read_sww_results(sww_file):
    """Read mesh, water depths, and momenta from an SWW file."""
    with Dataset(sww_file) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=float)
        y = np.asarray(ds.variables["y"][:], dtype=float)
        x = x + float(getattr(ds, "xllcorner", 0.0))
        y = y + float(getattr(ds, "yllcorner", 0.0))
        triangles = np.asarray(ds.variables["volumes"][:], dtype=int)
        times = np.asarray(ds.variables["time"][:], dtype=float)
        stage = np.asarray(ds.variables["stage"][:], dtype=float)
        elevation_values = np.asarray(ds.variables["elevation"][:], dtype=float)
        xmomentum = np.asarray(ds.variables["xmomentum"][:], dtype=float)
        ymomentum = np.asarray(ds.variables["ymomentum"][:], dtype=float)

    depths = np.maximum(stage - elevation_values[None, :], 0.0)
    return x, y, triangles, times, depths, xmomentum, ymomentum


def render_scalar_frame(
    triangulation,
    values,
    time_seconds,
    vmax,
    title_prefix,
    colorbar_label,
    cmap,
    show_mesh=False,
    overlay_model=None,
    x_origin=0.0,
    y_origin=0.0,
    show_osm_basemap=False,
    show_catchments=False,
    show_network=False,
    map_crs="EPSG:2326",
):
    """Render one scalar frame and return it as a base64 PNG string."""
    fig, ax = plt.subplots(figsize=(9.5, 3.9))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(triangulation.x)), float(np.max(triangulation.x)))
    ax.set_ylim(float(np.min(triangulation.y)), float(np.max(triangulation.y)))
    if show_osm_basemap:
        _draw_osm_basemap(ax, x_origin, y_origin, map_crs)
    if show_catchments:
        _draw_catchment_basemap(ax, overlay_model, x_origin, y_origin)

    levels = np.linspace(0.0, vmax, 31)
    contour = ax.tricontourf(
        triangulation,
        values,
        levels=levels,
        cmap=cmap,
        extend="max",
        alpha=0.74,
    )
    if show_mesh:
        ax.triplot(triangulation, color="black", linewidth=0.12, alpha=0.22)
    if show_network:
        _draw_network_overlay(ax, overlay_model, x_origin, y_origin)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title_prefix} at t = {time_seconds:.0f} s")
    cbar = fig.colorbar(contour, ax=ax, pad=0.015)
    cbar.set_label(colorbar_label)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_interactive_html(times, frame_images, output_file, page_title, image_alt):
    """Write a dependency-free HTML slider for browsing pre-rendered PNG frames."""
    frames_js = ",\n".join(f'"data:image/png;base64,{image}"' for image in frame_images)
    times_js = ", ".join(f"{time:.3f}" for time in times)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #17202a;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
      font-weight: 650;
    }}
    .viewer {{
      background: #ffffff;
      border: 1px solid #d8dee5;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(28, 39, 49, 0.08);
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #e2e6ea;
      border-radius: 6px;
      background: #ffffff;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
    }}
    button {{
      min-width: 84px;
      padding: 8px 12px;
      border: 1px solid #aeb8c2;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
      font-size: 14px;
      cursor: pointer;
    }}
    button:hover {{
      background: #f1f4f7;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .time {{
      min-width: 110px;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 14px;
      color: #34495e;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{page_title}</h1>
    <section class="viewer" aria-label="{page_title}">
      <img id="frame" src="" alt="{image_alt}">
      <div class="controls">
        <button id="play" type="button">Play</button>
        <input id="slider" type="range" min="0" max="{len(frame_images) - 1}" value="0" step="1">
        <div class="time" id="time"></div>
      </div>
    </section>
  </main>
  <script>
    const frames = [
{frames_js}
    ];
    const times = [{times_js}];
    const frame = document.getElementById("frame");
    const slider = document.getElementById("slider");
    const timeLabel = document.getElementById("time");
    const playButton = document.getElementById("play");
    let timer = null;

    function showFrame(index) {{
      const i = Math.max(0, Math.min(frames.length - 1, Number(index)));
      frame.src = frames[i];
      slider.value = i;
      timeLabel.textContent = `t = ${{times[i].toFixed(0)}} s`;
    }}

    function stopPlayback() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
      playButton.textContent = "Play";
    }}

    slider.addEventListener("input", (event) => {{
      stopPlayback();
      showFrame(event.target.value);
    }});

    playButton.addEventListener("click", () => {{
      if (timer !== null) {{
        stopPlayback();
        return;
      }}
      playButton.textContent = "Pause";
      timer = setInterval(() => {{
        const next = (Number(slider.value) + 1) % frames.length;
        showFrame(next);
      }}, 350);
    }});

    showFrame(0);
  </script>
</body>
</html>
"""
    output_file.write_text(html, encoding="utf-8")


def depth_color_limit(depths):
    """Use a stable color scale for all time steps."""
    vmax = float(np.nanpercentile(depths, 99.0))
    if vmax <= 0.0:
        vmax = float(np.nanmax(depths))
    if vmax <= 0.0:
        vmax = 1.0
    return vmax


def scalar_color_limit(values):
    """Use a stable color scale for non-negative scalar fields."""
    vmax = float(np.nanpercentile(values, 99.5))
    if vmax <= 0.0:
        vmax = float(np.nanmax(values))
    if vmax <= 0.0:
        vmax = 1.0
    return vmax


def mesh_edges(triangles):
    """Return unique mesh edges as vertex-index pairs."""
    edge_set = set()
    for a, b, c in triangles:
        edge_set.add(tuple(sorted((int(a), int(b)))))
        edge_set.add(tuple(sorted((int(b), int(c)))))
        edge_set.add(tuple(sorted((int(c), int(a)))))
    return np.asarray(sorted(edge_set), dtype=int)


def vertex_control_areas(triangulation):
    """Approximate nodal control areas by distributing triangle areas to vertices."""
    x = triangulation.x
    y = triangulation.y
    areas = np.zeros(len(x), dtype=float)
    for triangle in triangulation.triangles:
        a, b, c = [int(index) for index in triangle]
        tri_area = 0.5 * abs(
            (x[b] - x[a]) * (y[c] - y[a])
            - (x[c] - x[a]) * (y[b] - y[a])
        )
        share = tri_area / 3.0
        areas[a] += share
        areas[b] += share
        areas[c] += share
    return np.maximum(areas, 1.0e-9)


def diffuse_on_edges(values, edges, edge_lengths, diffusivity, dt):
    """Apply one explicit graph-diffusion step on mesh edges."""
    updated = values.copy()
    i = edges[:, 0]
    j = edges[:, 1]
    exchange = diffusivity * dt * (values[j] - values[i]) / np.maximum(
        edge_lengths * edge_lengths,
        1.0e-12,
    )
    np.add.at(updated, i, exchange)
    np.add.at(updated, j, -exchange)
    return np.maximum(updated, 0.0)


def apply_reaction_terms(concentration, dt, species):
    """Apply species-specific local reaction terms."""
    equation_name = species.get("equation", "conservative")
    if equation_name not in REACTION_EQUATIONS:
        raise ValueError(f"Unknown water-quality equation: {equation_name}")
    return REACTION_EQUATIONS[equation_name](concentration, dt, species)


def simulate_water_quality_species(
    times,
    depths,
    xmomentum,
    ymomentum,
    triangulation,
    species,
):
    """
    Simulate one water-quality species after the ANUGA hydrodynamics.

    This is a demo post-processor: concentration is transported by the SWW
    velocity field, diffused across mesh edges, released from a fixed source,
    and updated by species-specific local reaction equations.
    """
    x = triangulation.x
    y = triangulation.y
    coords = np.column_stack((x, y))
    edges = mesh_edges(triangulation.triangles)
    edge_lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    min_edge = float(np.min(edge_lengths))

    source_distance2 = (x - species["source_x"]) ** 2 + (y - species["source_y"]) ** 2
    source_shape = np.exp(-source_distance2 / (2.0 * species["source_radius"] ** 2))
    source_shape = source_shape / np.maximum(float(np.max(source_shape)), 1.0e-12)
    surface_sources = []
    for source in species.get("surface_sources", []):
        source_distance2 = (x - source["x"]) ** 2 + (y - source["y"]) ** 2
        shape = np.exp(-source_distance2 / (2.0 * source["radius"] ** 2))
        shape = shape / np.maximum(float(np.sum(shape)), 1.0e-12)
        surface_sources.append({**source, "shape": shape})
    node_areas = vertex_control_areas(triangulation)

    concentration = np.full(
        len(x),
        float(species.get("initial_background", 0.0)),
        dtype=float,
    )
    concentration_history = np.zeros((len(times), len(x)), dtype=float)
    concentration_history[0] = concentration

    for step in range(1, len(times)):
        dt_total = float(times[step] - times[step - 1])
        depth = depths[step - 1]
        wet = depth > MIN_WET_DEPTH
        u = np.divide(
            xmomentum[step - 1],
            depth,
            out=np.zeros_like(depth),
            where=wet,
        )
        v = np.divide(
            ymomentum[step - 1],
            depth,
            out=np.zeros_like(depth),
            where=wet,
        )

        max_speed = float(np.nanmax(np.hypot(u, v)))
        advective_dt = min_edge / max(max_speed, 1.0e-6)
        diffusivity = float(species.get("diffusivity", 0.0))
        diffusive_dt = 0.2 * min_edge * min_edge / max(diffusivity, 1.0e-12)
        stable_dt = max(min(advective_dt, diffusive_dt, dt_total), 1.0e-6)
        substeps = max(1, int(np.ceil(dt_total / stable_dt)))
        dt = dt_total / substeps

        for _ in range(substeps):
            interpolator = mtri.LinearTriInterpolator(triangulation, concentration)
            upstream_x = x - u * dt
            upstream_y = y - v * dt
            advected = interpolator(upstream_x, upstream_y)
            concentration = np.asarray(advected.filled(concentration), dtype=float)

            concentration = diffuse_on_edges(
                concentration,
                edges,
                edge_lengths,
                diffusivity,
                dt,
            )

            if times[step - 1] <= float(species["release_duration"]):
                concentration += float(species["source_rate"]) * source_shape * dt

            if surface_sources:
                source_index = min(step - 1, len(times) - 1)
                for source in surface_sources:
                    flow_series = source["flow_m3s"]
                    concentration_series = source["concentration"]
                    if source_index >= len(flow_series) or source_index >= len(concentration_series):
                        continue
                    source_flow = max(float(flow_series[source_index]), 0.0)
                    source_concentration = max(float(concentration_series[source_index]), 0.0)
                    if source_flow <= 0.0 or source_concentration <= 0.0:
                        continue
                    added_volume = source_flow * dt * source["shape"]
                    water_volume = np.maximum(depth * node_areas, 0.0)
                    concentration = np.divide(
                        concentration * water_volume + source_concentration * added_volume,
                        water_volume + added_volume,
                        out=concentration,
                        where=(water_volume + added_volume) > 1.0e-12,
                    )

            concentration = apply_reaction_terms(concentration, dt, species)
            concentration = np.where(wet, concentration, 0.0)

        concentration_history[step] = concentration

    return concentration_history


def simulate_water_quality(
    times,
    depths,
    xmomentum,
    ymomentum,
    triangulation,
    species_configs=None,
):
    """Simulate all configured water-quality species."""
    species_configs = species_configs if species_configs is not None else WATER_QUALITY_SPECIES
    return {
        species["key"]: simulate_water_quality_species(
            times,
            depths,
            xmomentum,
            ymomentum,
            triangulation,
            species,
        )
        for species in species_configs
    }


def show_combined_slider_plot(
    triangulation,
    times,
    datasets,
    overlay_model=None,
    x_origin=0.0,
    y_origin=0.0,
    show_osm_basemap=False,
    show_catchments=True,
    show_network=True,
    map_crs="EPSG:2326",
):
    """Create one Matplotlib window for selecting variables and time steps."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.2)

    selected = {"dataset": datasets[0]}
    scalar_layer = ax.tripcolor(
        triangulation,
        selected["dataset"]["values"][0],
        shading="gouraud",
        cmap=selected["dataset"]["cmap"],
        vmin=0.0,
        vmax=selected["dataset"]["vmax"],
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(triangulation.x)), float(np.max(triangulation.x)))
    ax.set_ylim(float(np.min(triangulation.y)), float(np.max(triangulation.y)))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if show_osm_basemap:
        _draw_osm_basemap(ax, x_origin, y_origin, map_crs)
    if show_catchments:
        _draw_catchment_basemap(ax, overlay_model, x_origin, y_origin)
    ax.triplot(triangulation, color="black", linewidth=0.12, alpha=0.18)
    if show_network:
        _draw_network_overlay(ax, overlay_model, x_origin, y_origin)

    cbar = fig.colorbar(
        scalar_layer,
        ax=ax,
        pad=0.015,
        label=selected["dataset"]["label"],
    )

    slider_ax = fig.add_axes([0.14, 0.07, 0.62, 0.035])
    slider = Slider(
        slider_ax,
        "Time step",
        0,
        len(times) - 1,
        valinit=0,
        valstep=1,
    )

    selector_ax = fig.add_axes([0.81, 0.45, 0.17, 0.32])
    selector_ax.set_title("Variable", fontsize=10)
    variable_labels = [dataset["menu_label"] for dataset in datasets]
    variable_selector = RadioButtons(selector_ax, variable_labels)
    state = {"pending_index": 0, "drawn_index": None}

    def draw_frame(index=None):
        dataset = selected["dataset"]
        if index is None:
            index = int(round(slider.val))
        index = max(0, min(len(times) - 1, int(index)))
        scalar_layer.set_array(dataset["values"][index])
        scalar_layer.set_cmap(dataset["cmap"])
        scalar_layer.set_clim(0.0, dataset["vmax"])
        cbar.update_normal(scalar_layer)
        cbar.set_label(dataset["label"])
        ax.set_title(f"{dataset['title']} at t = {times[index]:.0f} s")
        state["drawn_index"] = index
        fig.canvas.draw_idle()

    def update_time(value):
        state["pending_index"] = int(round(value))

    def commit_pending(_event=None):
        index = max(0, min(len(times) - 1, int(state["pending_index"])))
        if int(round(slider.val)) != index:
            slider.set_val(index)
            return
        if state["drawn_index"] != index:
            draw_frame(index)

    def update_variable(label):
        selected["dataset"] = datasets[variable_labels.index(label)]
        draw_frame(state["pending_index"])

    def update_key(event):
        if event.key not in ("left", "right"):
            return
        step = -1 if event.key == "left" else 1
        index = max(0, min(len(times) - 1, int(round(slider.val)) + step))
        state["pending_index"] = index
        slider.set_val(index)
        draw_frame(index)

    slider.on_changed(update_time)
    variable_selector.on_clicked(update_variable)
    fig.canvas.mpl_connect("button_release_event", commit_pending)
    fig.canvas.mpl_connect("key_press_event", update_key)
    draw_frame(0)

    fig._time_slider = slider
    fig._variable_selector = variable_selector
    return fig


def build_visual_outputs(show_plot=True):
    """Create depth and water-quality PNG/HTML viewers, plus optional slider plots."""
    x, y, triangles, times, depths, xmomentum, ymomentum = read_sww_results(SWW_FILE)
    triangulation = mtri.Triangulation(x, y, triangles)

    depth_vmax = depth_color_limit(depths)

    final_frame = render_scalar_frame(
        triangulation,
        depths[-1],
        times[-1],
        depth_vmax,
        "Flood depth",
        "Water depth (m)",
        "Blues",
        show_mesh=True,
    )
    FINAL_PNG.write_bytes(base64.b64decode(final_frame))

    frame_images = [
        render_scalar_frame(
            triangulation,
            depth,
            time_seconds,
            depth_vmax,
            "Flood depth",
            "Water depth (m)",
            "Blues",
        )
        for time_seconds, depth in zip(times, depths)
    ]
    write_interactive_html(
        times,
        frame_images,
        HTML_VIEWER,
        "ANUGA flood depth through time",
        "2D flood depth map",
    )

    water_quality = simulate_water_quality(
        times,
        depths,
        xmomentum,
        ymomentum,
        triangulation,
    )

    print(f"Final PNG: {FINAL_PNG}")
    print(f"Interactive HTML viewer: {HTML_VIEWER}")

    plot_datasets = [
        {
            "menu_label": "Flood depth",
            "title": "Flood depth",
            "label": "Water depth (m)",
            "cmap": "Blues",
            "values": depths,
            "vmax": depth_vmax,
        }
    ]

    for species in WATER_QUALITY_SPECIES:
        values = water_quality[species["key"]]
        species_vmax = scalar_color_limit(values)
        final_png = OUTDIR / f"final_{species['key']}_concentration.png"
        html_viewer = OUTDIR / f"{species['key']}_timeseries.html"

        species_final_frame = render_scalar_frame(
            triangulation,
            values[-1],
            times[-1],
            species_vmax,
            species["title"],
            species["label"],
            species["cmap"],
            show_mesh=True,
        )
        final_png.write_bytes(base64.b64decode(species_final_frame))

        species_frame_images = [
            render_scalar_frame(
                triangulation,
                concentration,
                time_seconds,
                species_vmax,
                species["title"],
                species["label"],
                species["cmap"],
            )
            for time_seconds, concentration in zip(times, values)
        ]
        write_interactive_html(
            times,
            species_frame_images,
            html_viewer,
            f"{species['title']} through time",
            f"2D {species['title']} map",
        )
        print(f"{species['title']} PNG: {final_png}")
        print(f"{species['title']} HTML viewer: {html_viewer}")

        plot_datasets.append(
            {
                "menu_label": species.get("menu_label", species["key"]),
                "title": species["title"],
                "label": species["label"],
                "cmap": species["cmap"],
                "values": values,
                "vmax": species_vmax,
            }
        )

    if show_plot:
        show_combined_slider_plot(triangulation, times, plot_datasets)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ANUGA demo and create a time-slider flood-depth viewer."
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Only build visual outputs from the existing simple_flood.sww file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate PNG and HTML outputs without opening the Matplotlib slider.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_simulation:
        if not SWW_FILE.exists():
            raise SystemExit(f"Cannot find existing SWW file: {SWW_FILE}")
    else:
        run_simulation()

    build_visual_outputs(show_plot=not args.no_show)


if __name__ == "__main__":
    main()
