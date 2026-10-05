# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
import pytest

from anemoi.transform.grids import UnstructuredGridFieldList
from anemoi.transform.grids import grid_registry
from anemoi.transform.grids.named import lookup
from anemoi.transform.grids.transverse_mercator import TransverseMercatorGrid

latitude_url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_extpar_0026_R03B07_G_20150805.g2"
tlat = "tlat"

longitudes_url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_extpar_0026_R03B07_G_20150805.g2"
tlon = "tlon"


def do_not_test_unstructured_from_url() -> None:
    """Test the UnstructuredGridFieldList class for loading data from URLs.

    Tests:
    - Loading latitude and longitude data from URLs.
    - Asserting the loaded data has the correct number of grid points.
    - Creating forcings from the loaded data and asserting their properties.
    """
    ds = UnstructuredGridFieldList.from_grib(latitude_url, longitudes_url, tlat, tlon)

    assert len(ds) == 1

    lats, lons = ds[0].grid_points()

    assert len(lats) == len(lons)

    forcings = ekd.from_source(
        "forcings",
        ds,
        date="2015-08-05",
        param=["cos_latitude", "sin_latitude"],
    )

    assert len(forcings) == 2


def test_lookup_o96() -> None:
    """Test the grids function for the 'o96' grid."""
    x = lookup("o96")
    assert x["latitudes"].mean() == pytest.approx(0.0)
    assert x["longitudes"].mean() == pytest.approx(179.14285714285714)
    assert x["latitudes"].shape == (40320,)
    assert x["longitudes"].shape == (40320,)
    assert x["latitudes"][31415] == pytest.approx(-31.324557701757268)
    assert x["longitudes"][31415] == pytest.approx(224.32835820895522)


def test_transverse_mercator_shape() -> None:
    """Test that the transverse Mercator grid produces flat lat/lon arrays of the expected size."""
    grid = TransverseMercatorGrid()
    lat, lon = grid.latlon()
    assert lat.shape == (grid.ni * grid.nj,)
    assert lon.shape == (grid.ni * grid.nj,)
    assert lat.shape == (385792,)


def test_transverse_mercator_origin() -> None:
    """Test that the grid point at (x=0, y=0) maps to the projection reference point.

    With the default UKV configuration, x = 0 is column i = 119 and y = 0 is row
    j = 611, so the reference point (latitude_0, longitude_0) lies on the grid.
    """
    grid = TransverseMercatorGrid()
    lat, lon = grid.latlon()
    i = int(round((0.0 - grid.x1) / grid.dx))
    j = int(round((grid.y1 - 0.0) / grid.dy))
    k = j * grid.ni + i
    assert lat[k] == pytest.approx(grid.latitude_0, abs=1e-9)
    assert lon[k] == pytest.approx(grid.longitude_0, abs=1e-9)


def test_transverse_mercator_corners() -> None:
    """Test the lat/lon of the four domain corners against known reference values.

    The expected values were computed independently with pyproj during the
    verification of the Met Office UKV GRIB2 example file (transverse Mercator
    grid definition template 12, 548 x 704 points at 2 km).
    """
    grid = TransverseMercatorGrid()
    lat, lon = grid.latlon()
    ni, nj = grid.ni, grid.nj
    # name: (i, j, expected lat, expected lon)
    corners = {
        "north-west": (0, 0, 59.92021488559434, -6.27229356342617),
        "north-east": (ni - 1, 0, 59.11228746191059, 13.07913260647756),
        "south-west": (0, nj - 1, 47.30196055266772, -5.156299365980432),
        "south-east": (ni - 1, nj - 1, 46.79110892558979, 9.248359053596639),
    }
    for name, (i, j, expected_lat, expected_lon) in corners.items():
        k = j * ni + i
        assert lat[k] == pytest.approx(expected_lat, abs=1e-6), name
        assert lon[k] == pytest.approx(expected_lon, abs=1e-6), name


def test_transverse_mercator_landmarks() -> None:
    """Test that known UK landmarks are located at the expected lat/lon.

    The flat indices are the nearest grid points to central London (51.503N,
    0.122W) and central Edinburgh (55.955N, 3.189W); the expected values were
    computed independently with pyproj during the verification of the Met
    Office UKV GRIB2 example file.
    """
    grid = TransverseMercatorGrid()
    lat, lon = grid.latlon()
    # name: (k, expected lat, expected lon)
    landmarks = {
        "london": (258292, 51.50301063982756, -0.12181078246079353),
        "edinburgh": (122834, 55.95477763416637, -3.188702800067505),
    }
    for name, (k, expected_lat, expected_lon) in landmarks.items():
        assert lat[k] == pytest.approx(expected_lat, abs=1e-6), name
        assert lon[k] == pytest.approx(expected_lon, abs=1e-6), name


def test_transverse_mercator_from_config() -> None:
    """Test that the grid is created from a configuration via the grid registry."""
    grid = grid_registry.from_config({"transverse_mercator": {}})
    assert isinstance(grid, TransverseMercatorGrid)
    lat, lon = grid.latlon()
    assert lat.shape == (548 * 704,)


def test_transverse_mercator_validation() -> None:
    """Test that invalid grid parameters are rejected."""
    with pytest.raises(ValueError):
        TransverseMercatorGrid(latitude_0=91.0)
    with pytest.raises(ValueError):
        TransverseMercatorGrid(ni=0)
    with pytest.raises(ValueError):
        TransverseMercatorGrid(dx=-1.0)


if __name__ == "__main__":
    from anemoi.utils.testing import run_tests

    run_tests(globals())
