# Charun–Buni red belt, Upper Chitral

Terrain and mineral-prospectivity analysis of the Mastuj valley between Reshun and Buni, Upper Chitral, Pakistan (36.11–36.34°N, 72.05–72.32°E). Uses public satellite archives: Copernicus DEM, Sentinel-2, ASTER, NASA EMIT, and Sentinel-1.

**Live report:** https://dinarwali4.github.io/charun-buni-red-belt/

## Findings
- The maroon bands are the **Reshun Formation**: hematite-red siltstone and conglomerate of Late Cretaceous to Palaeogene age. They form a 25 km ENE belt of about 44 km², and its margins are thrust faults (Stauffer 1975, USGS OFR 75-556).
- **No multispectral gossan is visible.** 68% of the area is exposed rock. Apparent anomalies above 4,500 m disappear when normalised for elevation.
- **Two weak EMIT leads, neither confirmed by a second sensor:**
  - E1: jarosite/copiapite at 36.2090°N 72.2700°E.
  - E2: an "azurite" scatter near Buni.
- **The potential is structural:** fault-hosted Sb–Pb–Au(–Ag) veins along the Reshun fault zone, as at Krinj and Awireth. The recommended next step is stream-sediment sampling at 14 sites (`data/stream_sites_final.json`).

No figure here is an assay. Grade and tonnage require ground sampling.

## Contents
| Path | What |
|---|---|
| `docs/` | The published report (GitHub Pages) and `Charun_Buni_Prospectivity.kml` |
| `figures/` | Full-resolution maps and charts |
| `data/` | JSON results: colour units, anomaly clusters, EMIT persistence, SAR and DEM statistics, sampling sites |
| `scripts/` | The full Python pipeline |

## Reproducing
Python 3.13 with rasterio, pystac-client, planetary-computer, netCDF4, scikit-learn, scikit-image, scipy, pyproj, matplotlib.

Run in this order:
1. Fetch data: `s2_fetch.py`, `dem_proc.py`, `s1_fetch2.py`, `get_links.py`, then `dl.sh`.
2. Process: `emit_proc.py`, `emit_persist.py`, `aster_tir.py`, `aster_swir.py`, `s2_analysis.py`, `dem_analysis.py`.
3. Integrate: `integrate.py`, `anom2.py`, `drainage.py`, `sar.py`.
4. Figures and outputs: `figures.py`, `make_kml.py`.

EMIT and ASTER downloads need a NASA Earthdata bearer token in `~/.edl_token`. Raw rasters (~1 GB) are not committed.

## Sources
- Stauffer, K.W. (1975). [Reconnaissance geology of the central Mastuj Valley, Chitral State, Pakistan](https://pubs.usgs.gov/publication/ofr75556). USGS OFR 75-556.
- DG Mines & Minerals KP (2014). [Follow-up Exploration Targets, District Chitral](https://kpminerals.gov.pk/uploads/geological_reports/attachment/1655712609_Follow-up%20Exploration%20Targets%20(Metals%20and%20Gemstones)%20Chitral.pdf).
- Data: Copernicus GLO-30 (ESA), Sentinel-1/2 (ESA Copernicus), ASTER and EMIT (NASA LP DAAC), OpenStreetMap place names (© OpenStreetMap contributors).
- The screenshot in `docs/img/` contains Mapbox/Maxar imagery (© Mapbox, © Maxar), used for annotation.

Prepared by RegenX.eco, September 2026.
