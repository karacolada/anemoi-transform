# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# or does not submit to any jurisdiction.


"""Transverse Mercator grid (Met Office UKV, GRIB2 grid definition template 12).

The grid is a transverse Mercator projection on a sphere. The projected
coordinates of the grid points are defined by

    x(i) = x1 + dx * i      (i = 0 .. ni - 1, west to east)
    y(j) = y1 - dy * j      (j = 0 .. nj - 1, north to south)

so that row ``j = 0`` is the northernmost row, matching the Met Office UKV
scanning convention (``jScansPositively`` with ``Y1 > Y2``). The flat lat/lon
arrays returned by :meth:`TransverseMercatorGrid.latlon` follow the GRIB2
packing order in which the i index varies fastest, i.e. flat index
``k = j * ni + i``.

The projection origin ``(x=0, y=0)`` is the reference point
``(latitude_0, longitude_0)``. The constructor arguments default to the
production UKV configuration:

    latitude_0 = 49.0, longitude_0 = -2.0,
    ni = 548, nj = 704, dx = dy = 2000 m,
    x1 = -238000 m, y1 = 1222000 m.

The earth radius defaults to 6371229 m, the spherical earth used by the UM
model. (The GRIB2 ``shapeOfTheEarth = 3`` field of the UKV files refers to the
WMO reference sphere of 6371200 m; the two radii differ by less than 10 m in
the resulting lat/lon anywhere over the UKV domain, so the difference is
immaterial. The radius is parameterised so that either can be used.)
"""

import logging
from functools import lru_cache

import numpy as np
from pyproj import CRS
from pyproj import Transformer

from . import Grid
from . import grid_registry

LOG = logging.getLogger(__name__)

# WGS84 geographic CRS, given with explicit semi-axes so that no PROJ
# database is required.
_LONGLAT_PROJ4 = "+proj=longlat +a=6378137 +b=6356752.314245 +no_defs"


def _transform_to_latlon(proj4: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform projected coordinates to WGS84 latitudes and longitudes.

    Parameters
    ----------
    proj4 : str
        PROJ4 string describing the projected coordinate system.
    x : np.ndarray
        Easting coordinates in metres (flat array).
    y : np.ndarray
        Northing coordinates in metres (flat array, same shape as ``x``).

    Returns
    -------
    tuple of np.ndarray
        ``(latitudes, longitudes)`` in degrees, same shape as the inputs.
    """
    source = CRS.from_proj4(proj4)
    target = CRS.from_proj4(_LONGLAT_PROJ4)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


@lru_cache(1)
def transverse_mercator_grid(
    latitude_0: float,
    longitude_0: float,
    ni: int,
    nj: int,
    dx: float,
    dy: float,
    x1: float,
    y1: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the latitudes and longitudes of a transverse Mercator grid.

    Parameters
    ----------
    latitude_0 : float
        Latitude of the projection reference point in degrees.
    longitude_0 : float
        Longitude of the projection reference point in degrees.
    ni : int
        Number of grid points in the i (east-west) direction.
    nj : int
        Number of grid points in the j (north-south) direction.
    dx : float
        Grid spacing in the i direction in metres.
    dy : float
        Grid spacing in the j direction in metres.
    x1 : float
        Easting of the westernmost column in metres.
    y1 : float
        Northing of the northernmost row in metres.
    radius : float
        Radius of the spherical earth in metres.

    Returns
    -------
    tuple of np.ndarray
        Flat ``(latitudes, longitudes)`` arrays in degrees of length
        ``ni * nj``; flat index ``k = j * ni + i`` with row ``j = 0``
        the northernmost row.
    """
    x = x1 + dx * np.arange(ni, dtype=np.float64)
    y = y1 - dy * np.arange(nj, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    proj4 = (
        f"+proj=tmerc +lat_0={latitude_0:g} +lon_0={longitude_0:g} +k=1"
        f" +x_0=0 +y_0=0 +R={radius:g} +no_defs"
    )
    lat, lon = _transform_to_latlon(proj4, xx.ravel(), yy.ravel())
    return lat, lon


@grid_registry.register("transverse_mercator")
class TransverseMercatorGrid(Grid):
    """Transverse Mercator grid, e.g. the Met Office UKV model grid.

    The grid definition mirrors the Met Office GRIB2 grid definition
    template 12 ("transverse_mercator"). See the module docstring for the
    coordinate conventions and the mapping to the GRIB2 keys.
    """

    def __init__(
        self,
        latitude_0: float = 49.0,
        longitude_0: float = -2.0,
        ni: int = 548,
        nj: int = 704,
        dx: float = 2000.0,
        dy: float = 2000.0,
        x1: float = -238000.0,
        y1: float = 1222000.0,
        radius: float = 6371229.0,
    ) -> None:
        """Initialise the transverse Mercator grid.

        Parameters
        ----------
        latitude_0 : float, optional
            Latitude of the projection reference point in degrees, by default 49.0.
        longitude_0 : float, optional
            Longitude of the projection reference point in degrees, by default -2.0.
        ni : int, optional
            Number of grid points in the i direction, by default 548.
        nj : int, optional
            Number of grid points in the j direction, by default 704.
        dx : float, optional
            Grid spacing in the i direction in metres, by default 2000.0.
        dy : float, optional
            Grid spacing in the j direction in metres, by default 2000.0.
        x1 : float, optional
            Easting of the westernmost column in metres, by default -238000.0.
        y1 : float, optional
            Northing of the northernmost row in metres, by default 1222000.0.
        radius : float, optional
            Radius of the spherical earth in metres, by default 6371229.0.
        """
        self._validate(latitude_0=latitude_0, longitude_0=longitude_0)
        self._validate(ni=ni, nj=nj, dx=dx, dy=dy, radius=radius)
        self.latitude_0 = float(latitude_0)
        self.longitude_0 = float(longitude_0)
        self.ni = int(ni)
        self.nj = int(nj)
        self.dx = float(dx)
        self.dy = float(dy)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.radius = float(radius)

    @staticmethod
    def _validate(**kwargs: float | int) -> None:
        """Validate grid parameters.

        Parameters
        ----------
        **kwargs
            Parameter name/value pairs to validate.

        Raises
        ------
        ValueError
            If any parameter is outside its valid range.
        """
        for name, value in kwargs.items():
            if name in ("latitude_0",):
                if not -90.0 <= float(value) <= 90.0:
                    raise ValueError(f"latitude_0 must be in [-90, 90], got {value}")
            elif name in ("longitude_0",):
                if not -180.0 <= float(value) <= 180.0:
                    raise ValueError(f"longitude_0 must be in [-180, 180], got {value}")
            elif name in ("ni", "nj"):
                if int(value) < 1:
                    raise ValueError(f"{name} must be a positive integer, got {value}")
            else:
                if float(value) <= 0.0:
                    raise ValueError(f"{name} must be positive, got {value}")

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the latitudes and longitudes of the grid.

        Returns
        -------
        tuple of np.ndarray
            Flat ``(latitudes, longitudes)`` arrays in degrees of length
            ``ni * nj``; flat index ``k = j * ni + i`` with row ``j = 0``
            the northernmost row.
        """
        return transverse_mercator_grid(
            self.latitude_0,
            self.longitude_0,
            self.ni,
            self.nj,
            self.dx,
            self.dy,
            self.x1,
            self.y1,
            self.radius,
        )
