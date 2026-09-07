"""Read rectangular regions of an image that does not fit in memory.

Slicing normally decodes the whole source once and hands every slice a numpy view
into it, which costs nothing extra. That trade stops working when the decode alone
is larger than RAM, as it is for a whole-slide scan. A `LazyImageSource` replaces
the decoded array with an object that reads one region at a time, so slicing holds
only the slices it is currently working on.

Implementations index like the array they stand in for::

    region = source[top:bottom, left:right]

which is what `slice_image` already does, so the same code path serves both.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from sahi.logger import logger

# Decoding beyond this costs more memory than a slicing run should assume it has, so a
# tiled TIFF this large is read in regions instead. Tuned to be well clear of the images
# sahi is normally pointed at, so that ordinary use is never diverted onto the lazy path.
LARGE_IMAGE_THRESHOLD_BYTES = 2_000_000_000


class LazyImageSource(ABC):
    """An image whose regions are read on demand instead of decoded up front.

    Subclasses implement `shape` and `read_region`. Indexing with a pair of slices
    is provided on top of them so a source can stand in for the decoded array.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the image shape as (height, width, channels)."""

    @abstractmethod
    def read_region(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Return the given region as an HWC uint8 array.

        Args:
            left: X coordinate of the region's left edge.
            top: Y coordinate of the region's top edge.
            width: Region width in pixels.
            height: Region height in pixels.

        Returns:
            numpy.ndarray: The region, as an HWC array.
        """

    def __getitem__(self, key: Any) -> np.ndarray:
        """Read a region using `source[top:bottom, left:right]` array syntax."""
        if not isinstance(key, tuple) or len(key) not in (2, 3):
            raise TypeError(
                f"{type(self).__name__} supports indexing by [rows, cols] slices, got {key!r}. "
                "Use read_region() for anything else."
            )
        rows, cols = key[0], key[1]
        if not isinstance(rows, slice) or not isinstance(cols, slice):
            raise TypeError(f"{type(self).__name__} row and column indices must both be slices, got {key!r}.")
        if len(key) == 3 and key[2] not in (slice(None), Ellipsis):
            raise TypeError(f"{type(self).__name__} cannot select a channel subset, got {key[2]!r}.")

        height, width = self.shape[:2]
        top, bottom, row_step = rows.indices(height)
        left, right, col_step = cols.indices(width)
        if row_step != 1 or col_step != 1:
            raise TypeError(f"{type(self).__name__} does not support strided indexing, got {key!r}.")

        return self.read_region(left=left, top=top, width=max(0, right - left), height=max(0, bottom - top))

    @property
    def ndim(self) -> int:
        """Number of array dimensions, mirroring `numpy.ndarray.ndim`."""
        return len(self.shape)

    def close(self) -> None:
        """Release any handle the source holds. Overridden when there is one."""

    def __enter__(self) -> LazyImageSource:
        """Enter a context that closes the source on exit."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the source."""
        self.close()


def is_lazy_image_source(image: Any) -> bool:
    """Return whether `image` reads regions on demand rather than decoding up front."""
    return isinstance(image, LazyImageSource)


class SlideImageSource(LazyImageSource):
    """Region reads backed by an already-open whole-slide image handle.

    Wraps any object exposing the OpenSlide reader interface -- `openslide.OpenSlide`,
    `tiffslide.TiffSlide` and `cucim.CuImage` all qualify -- so the backend stays the
    caller's choice and no slide library becomes a dependency of sahi.

    Args:
        slide: An open slide handle with `read_region` and `level_dimensions`.
        level: Pyramid level to read from. Level 0 is full resolution.

    Example:
        >>> import tiffslide  # doctest: +SKIP
        >>> from sahi.utils.lazy_image import SlideImageSource  # doctest: +SKIP
        >>> source = SlideImageSource(tiffslide.TiffSlide("scan.tiff"))  # doctest: +SKIP
    """

    def __init__(self, slide: Any, level: int = 0) -> None:
        """Initialize the source from an open slide handle."""
        self.slide = slide
        self.level = level
        width, height = slide.level_dimensions[level]
        self._shape = (int(height), int(width), 3)
        # read_region takes level 0 coordinates whatever level it reads, so a level
        # other than 0 needs its own coordinates scaled back up before the call
        self._downsample = float(slide.level_downsamples[level])

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the selected pyramid level as (height, width, 3)."""
        return self._shape

    def read_region(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Read a region of the selected level, dropping the alpha channel."""
        level0_left = round(left * self._downsample)
        level0_top = round(top * self._downsample)
        region = self.slide.read_region((level0_left, level0_top), self.level, (width, height))
        return np.asarray(region)[:, :, :3]

    def close(self) -> None:
        """Close the underlying slide handle."""
        self.slide.close()


class TiffFileImageSource(LazyImageSource):
    """Region reads backed by `tifffile` over a tiled (usually pyramidal) TIFF.

    Needs the file to be tiled, which pathology and remote-sensing scans are, so that
    a region touches only a few tiles. Requires `zarr`, which `tifffile` uses to map
    the tiles without decoding the rest of the level.

    Args:
        path: Path to the TIFF file.
        level: Pyramid level to read from. Level 0 is full resolution.
        series: Series index within the file.
    """

    def __init__(self, path: str, level: int = 0, series: int = 0) -> None:
        """Open the file and map the requested level."""
        try:
            import tifffile
        except ImportError:
            raise ImportError("TiffFileImageSource needs tifffile, run 'pip install tifffile zarr'.")
        try:
            import zarr
        except ImportError:
            raise ImportError("TiffFileImageSource needs zarr to read regions, run 'pip install zarr'.")

        self._tif = tifffile.TiffFile(path)
        store = self._tif.series[series].aszarr(level=level)
        self._store = store
        self._array = zarr.open(store, mode="r")
        shape = tuple(int(dim) for dim in self._array.shape)
        self._shape = shape if len(shape) == 3 else (shape[0], shape[1], 1)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the selected pyramid level as (height, width, channels)."""
        return self._shape

    def read_region(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Read a region, decoding only the TIFF tiles it covers."""
        region = np.asarray(self._array[top : top + height, left : left + width])
        if region.ndim == 2:
            region = region[:, :, np.newaxis]
        return region[:, :, :3]

    def close(self) -> None:
        """Close the zarr store and the TIFF handle."""
        self._store.close()
        self._tif.close()


def open_large_image_source(
    image: Any,
    threshold_bytes: int | None = None,
) -> LazyImageSource | None:
    """Return a lazy source for an image too large to decode, or None to decode normally.

    Lets a caller pass a path to a gigapixel scan and have it read in regions, without
    having to know that is what happened. Only a tiled TIFF qualifies: an untiled image
    has to be decoded from the start of the file whatever region is wanted, so reading it
    piecemeal would be slower than decoding it once.

    Args:
        image: The image about to be sliced. Anything that is not a local TIFF path is
            declined, so callers can pass this whatever they were given.
        threshold_bytes: Decoded size above which regions are read instead of decoding.
            Defaults to `LARGE_IMAGE_THRESHOLD_BYTES`, read at call time so it can be raised
            or lowered for a run.

    Returns:
        LazyImageSource | None: A source to read regions from, or None to decode as usual.
    """
    if threshold_bytes is None:
        threshold_bytes = LARGE_IMAGE_THRESHOLD_BYTES
    if is_lazy_image_source(image):
        return None
    if not isinstance(image, (str, os.PathLike)):
        return None

    path = os.fspath(image)
    if path.startswith("http") or Path(path).suffix.lower() not in (".tif", ".tiff"):
        return None

    try:
        import tifffile
    except ImportError:
        # nothing to read regions with, so the caller decodes as it always has
        return None

    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            shape = tuple(int(dim) for dim in series.shape)
            is_tiled = bool(series.pages[0].is_tiled)
    except Exception as error:
        logger.debug(f"could not inspect {path} for lazy reading ({error}), decoding it as usual")
        return None

    if len(shape) < 2:
        return None
    decoded_bytes = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
    if decoded_bytes < threshold_bytes:
        return None

    if not is_tiled:
        logger.warning(
            f"{path} needs {decoded_bytes / 1e9:.1f} GB to decode and is not tiled, so it cannot be read "
            "in regions. Decoding it whole, which may exhaust memory."
        )
        return None

    try:
        source = TiffFileImageSource(path)
    except ImportError:
        logger.warning(
            f"{path} needs {decoded_bytes / 1e9:.1f} GB to decode and could be read in regions instead, "
            "but that needs zarr. Run 'pip install sahi[wsi]'. Decoding it whole for now."
        )
        return None

    logger.info(
        f"{path} needs {decoded_bytes / 1e9:.1f} GB to decode, so it is being read in regions instead. "
        f"Full resolution is {source.shape[1]} x {source.shape[0]}."
    )
    return source
