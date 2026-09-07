"""Image slicing utilities for splitting large images into tiles."""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from shapely.errors import TopologicalError
from tqdm import tqdm

from sahi.annotation import BoundingBox, Mask
from sahi.logger import logger
from sahi.utils.coco import Coco, CocoAnnotation, CocoImage, create_coco_dict
from sahi.utils.cv import IMAGE_EXTENSIONS_LOSSY, read_image_as_pil
from sahi.utils.file import load_json, save_json
from sahi.utils.lazy_image import LazyImageSource, is_lazy_image_source, open_large_image_source

_CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(1, min(32, _CPU_COUNT * 2))


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int | None = None,
    slice_width: int | None = None,
    auto_slice_resolution: bool | None = True,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
) -> list[list[int]]:
    """Generate bounding boxes for slicing an image into crops.

    The function calculates the coordinates for each slice based on the provided
    image dimensions, slice size, and overlap ratios. If slice size is not provided
    and auto_slice_resolution is True, the function will automatically determine
    appropriate slice parameters.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these parameters from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        if overlap_height_ratio is not None and overlap_height_ratio >= 1.0:
            raise ValueError("Overlap ratio must be less than 1.0")
        if overlap_width_ratio is not None and overlap_width_ratio >= 1.0:
            raise ValueError("Overlap ratio must be less than 1.0")
        y_overlap = int((overlap_height_ratio if overlap_height_ratio is not None else 0.2) * slice_height)
        x_overlap = int((overlap_width_ratio if overlap_width_ratio is not None else 0.2) * slice_width)
    elif auto_slice_resolution:
        x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                x_max = min(image_width, x_max)
                y_max = min(image_height, y_max)
                x_min = max(0, x_max - slice_width)
                y_min = max(0, y_max - slice_height)
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def annotation_inside_slice(annotation: dict, slice_bbox: list[int]) -> bool:
    """Check whether annotation coordinates lie inside slice coordinates.

    Args:
        annotation (dict): Single annotation entry in COCO format.
        slice_bbox (List[int]): Generated from `get_slice_bboxes`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].

    Returns:
        (bool): True if any annotation coordinate lies inside slice.
    """
    left, top, width, height = annotation["bbox"]

    right = left + width
    bottom = top + height

    if left >= slice_bbox[2]:
        return False
    if top >= slice_bbox[3]:
        return False
    if right <= slice_bbox[0]:
        return False
    if bottom <= slice_bbox[1]:
        return False

    return True


def process_coco_annotations(
    coco_annotation_list: list[CocoAnnotation], slice_bbox: list[int], min_area_ratio: float
) -> list[CocoAnnotation]:
    """Slices and filters given list of CocoAnnotation objects with given 'slice_bbox' and 'min_area_ratio'.

    Args:
        coco_annotation_list: List[CocoAnnotation]
            Annotations to slice and filter.
        slice_bbox (List[int]): Generated from `get_slice_bboxes`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].
        min_area_ratio (float): If the cropped annotation area to original
            annotation ratio is smaller than this value, the annotation is
            filtered out. Default 0.1.

    Returns:
        (List[CocoAnnotation]): Sliced annotations.
    """
    sliced_coco_annotation_list: list[CocoAnnotation] = []
    for coco_annotation in coco_annotation_list:
        if annotation_inside_slice(coco_annotation.json, slice_bbox):
            sliced_coco_annotation = coco_annotation.get_sliced_coco_annotation(slice_bbox)
            if sliced_coco_annotation.area / coco_annotation.area >= min_area_ratio:
                sliced_coco_annotation_list.append(sliced_coco_annotation)
    return sliced_coco_annotation_list


class SlicedImage:
    """Container for a sliced image and its metadata."""

    def __init__(
        self,
        image: np.ndarray | None,
        coco_image: CocoImage,
        starting_pixel: list[int],
        source: Any | None = None,
        slice_bbox: list[int] | None = None,
    ) -> None:
        """Initialize SlicedImage.

        Args:
            image: np.array
                Sliced image. May be None when `source` and `slice_bbox` are given, in
                which case the pixels are read from the source on each access instead.
            coco_image: CocoImage
                Coco styled image object that belong to sliced image.
            starting_pixel: list of list of int
                Starting pixel coordinates of the sliced image.
            source: optional
                What to read the pixels from when `image` is None. Anything indexable as
                `source[top:bottom, left:right]`, such as a LazyImageSource.
            slice_bbox: list of int, optional
                This slice's [left, top, right, bottom] region within `source`.
        """
        self._image = image
        self.coco_image = coco_image
        self.starting_pixel = starting_pixel
        self._source = source
        self._slice_bbox = slice_bbox

    @property
    def image(self) -> np.ndarray:
        """The slice pixels.

        Read from the source on every access when the slice was created without them.
        The read is deliberately not cached: holding every slice read so far is the
        memory cost a lazy source exists to avoid.
        """
        if self._image is not None:
            return self._image
        if self._source is None or self._slice_bbox is None:
            raise ValueError("SlicedImage was given neither pixels nor a source to read them from.")
        left, top, right, bottom = self._slice_bbox
        return self._source[top:bottom, left:right]

    @image.setter
    def image(self, value: np.ndarray) -> None:
        """Set the slice pixels, replacing any deferred read."""
        self._image = value


class SliceImageResult:
    """Container for sliced image results."""

    def __init__(
        self,
        original_image_size: list[int],
        image_dir: str | None = None,
        original_image: np.ndarray | None = None,
        owned_source: LazyImageSource | None = None,
    ) -> None:
        """Initialize SliceImageResult.

        Args:
            image_dir: str
                Directory of the sliced image exports.
            original_image_size: list of int
                Size of the unsliced original image in [height, width].
            original_image: np.ndarray, optional
                The decoded source image. Every slice is a view into it, so it is
                alive for as long as this result is; holding it lets callers reuse
                the decode instead of reading the file again.
        """
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir
        self.original_image = original_image
        # a source opened on the caller's behalf is this result's to close. One the caller
        # opened is not, so it is never stored here.
        self._owned_source = owned_source

        self._sliced_image_list: list[SlicedImage] = []

    @property
    def is_lazy(self) -> bool:
        """Whether slices are read from a source on access rather than held in memory."""
        return bool(self._sliced_image_list) and self._sliced_image_list[0]._image is None

    def close(self) -> None:
        """Close the source this result opened, if it opened one.

        A source the caller passed in stays open: closing it is the caller's to do.
        """
        if self._owned_source is not None:
            self._owned_source.close()
            self._owned_source = None

    def __enter__(self) -> SliceImageResult:
        """Enter a context that closes any source this result opened."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close any source this result opened."""
        self.close()

    def add_sliced_image(self, sliced_image: SlicedImage) -> None:
        """Add a sliced image to the result."""
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self) -> list[SlicedImage]:
        """Return list of sliced images."""
        return self._sliced_image_list

    @property
    def images(self) -> list[np.ndarray]:
        """Returns sliced images.

        Returns:
            images: a list of np.array
        """
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def coco_images(self) -> list[CocoImage]:
        """Returns CocoImage representation of SliceImageResult.

        Returns:
            coco_images: a list of CocoImage
        """
        coco_images: list = []
        for sliced_image in self._sliced_image_list:
            coco_images.append(sliced_image.coco_image)
        return coco_images

    @property
    def starting_pixels(self) -> list[list[int]]:
        """Returns a list of starting pixels for each slice.

        Returns:
            starting_pixels: a list of starting pixel coords [x,y]
        """
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> list[str]:
        """Returns a list of filenames for each slice.

        Returns:
            filenames: a list of filenames as str
        """
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def slice_at(self, i: int) -> np.ndarray:
        """Return the pixels of slice `i`.

        Prefer this to `images[i]`. `images` rebuilds the list of every slice on each
        call, which is O(n) per access and, on a lazy source, reads the whole image.

        Args:
            i: Index of the slice.

        Returns:
            numpy.ndarray: The slice, as an HWC array.
        """
        return self._sliced_image_list[i].image

    def starting_pixel_at(self, i: int) -> list[int]:
        """Return the starting pixel of slice `i` without building the full list.

        Args:
            i: Index of the slice.

        Returns:
            list of int: The slice's starting pixel coordinates as [x, y].
        """
        return self._sliced_image_list[i].starting_pixel

    def __getitem__(self, i: int | slice | list | tuple) -> dict | list:
        """Get sliced image(s) by index or slice."""

        def _prepare_ith_dict(i: int) -> dict:
            return {
                "image": self.images[i],
                "coco_image": self.coco_images[i],
                "starting_pixel": self.starting_pixels[i],
                "filename": self.filenames[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self) -> int:
        """Return number of sliced images."""
        return len(self._sliced_image_list)


def _slice_file_suffix(image: str | Image.Image | np.ndarray, out_ext: str | None = None) -> str:
    """Resolve the file extension exported slices are written with.

    Takes over from the `image_pil.filename` lookup that slicing an array in place made
    impossible, and keeps its outcome exactly. Only an already-open PIL image carries a
    filename: `read_image_as_pil` returns a converted copy, so a path loses it and exports
    as png. Lossy sources become png too, so repeated slicing does not compound the
    compression.
    """
    if out_ext:
        return out_ext
    source_suffix = Path(str(getattr(image, "filename", ""))).suffix
    if not source_suffix or source_suffix in IMAGE_EXTENSIONS_LOSSY:
        return ".png"
    return source_suffix


def slice_image(
    image: str | Image.Image | np.ndarray | LazyImageSource,
    coco_annotation_list: list[CocoAnnotation] | None = None,
    output_file_name: str | None = None,
    output_dir: str | None = None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    auto_slice_resolution: bool | None = True,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
    auto_lazy: bool = True,
) -> SliceImageResult:
    """Slice a large image into smaller windows. If output_file_name and output_dir is given, export sliced images.

    Args:
        image (str or PIL.Image or np.ndarray or LazyImageSource): File path of image, Pillow Image,
            numpy array, or a LazyImageSource to be sliced. A LazyImageSource is never decoded
            whole: each slice is read from it on access, so an image larger than memory can be
            sliced by holding only the slices in use.
        coco_annotation_list (List[CocoAnnotation], optional): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.
        min_area_ratio (float, optional): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix for lossless image formats and png for lossy formats ('.jpg','.jpeg').
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.
        exif_fix (bool): Whether to apply an EXIF fix to the image.
        auto_lazy (bool): Whether a path to an image too large to decode may be read in
            regions instead. Only a tiled TIFF over `LARGE_IMAGE_THRESHOLD_BYTES` qualifies,
            so ordinary images are unaffected. Set False to always decode. Default True.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
    """
    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    def _export_single_slice(slice_index: int, output_dir: str, slice_file_name: str) -> None:
        # the slice is read inside the worker so a lazy source holds only the slices
        # being written rather than all of them at once
        image_pil = read_image_as_pil(sliced_image_result.slice_at(slice_index), exif_fix=exif_fix)
        slice_file_path = str(Path(output_dir) / slice_file_name)
        # export sliced image
        image_pil.save(slice_file_path)
        image_pil.close()  # to fix https://github.com/obss/sahi/issues/565
        verboselog("sliced image path: " + slice_file_path)

    # create outdir if not present
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # a path to an image too large to decode is read in regions instead, so that callers
    # can pass one without having to know that is what happened
    owned_source = open_large_image_source(image) if auto_lazy else None
    if owned_source is not None:
        image = owned_source

    # a lazy source is already what slices are read from, so there is nothing to decode
    # and no full-size array for the result to hold on to
    slice_source_is_lazy = is_lazy_image_source(image)
    image_arr: np.ndarray | LazyImageSource
    if slice_source_is_lazy:
        image_arr = image  # type: ignore[assignment]
    else:
        # read as an array, so no full-size PIL copy is held alongside it
        image_arr = read_image_as_pil(image, exif_fix=exif_fix, return_arr=True)  # type: ignore[assignment]
    image_height, image_width = image_arr.shape[:2]
    verboselog("image.shape: " + str((image_width, image_height)))

    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {(image_width, image_height)} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(
        original_image_size=[image_height, image_width],
        image_dir=output_dir,
        # a lazy source has no decoded image to reuse, and materializing one here would
        # defeat the point of slicing it lazily
        original_image=None if slice_source_is_lazy else image_arr,  # type: ignore[arg-type]
        owned_source=owned_source,
    )

    suffix = _slice_file_suffix(image, out_ext)

    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        # an in-memory slice is a free view into the decode, so it is taken now as before.
        # a lazy one is left to be read on access, so slices do not accumulate.
        image_pil_slice = None if slice_source_is_lazy else image_arr[tly:bry, tlx:brx]

        # set image file name and path
        slice_suffixes = "_".join(map(str, slice_bbox))
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"

        # create coco image
        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]
        coco_image = CocoImage(file_name=slice_file_name, height=slice_height, width=slice_width)

        # append coco annotations (if present) to coco image
        if coco_annotation_list is not None:
            min_area_ratio_val: float = min_area_ratio if min_area_ratio is not None else 0.1
            for sliced_coco_annotation in process_coco_annotations(
                coco_annotation_list, slice_bbox, min_area_ratio_val
            ):
                coco_image.add_annotation(sliced_coco_annotation)

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice,
            coco_image=coco_image,
            starting_pixel=[slice_bbox[0], slice_bbox[1]],
            source=image_arr,
            slice_bbox=[tlx, tly, brx, bry],
        )
        sliced_image_result.add_sliced_image(sliced_image)

    # export slices if output directory is provided
    if output_file_name and output_dir:
        # Use a context-managed ThreadPoolExecutor for clean shutdown and
        # limit workers based on CPU count to avoid oversubscription.
        max_workers = min(MAX_WORKERS, len(sliced_image_result))
        max_workers = max(1, max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map will schedule tasks and wait for completion when the context exits
            list(
                executor.map(
                    _export_single_slice,
                    range(len(sliced_image_result)),
                    [output_dir] * len(sliced_image_result),
                    sliced_image_result.filenames,
                )
            )

    verboselog(
        "Num slices: " + str(n_ims) + " slice_height: " + str(slice_height) + " slice_width: " + str(slice_width)
    )

    return sliced_image_result


def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str,
    output_dir: str | None = None,
    ignore_negative_samples: bool | None = False,
    slice_height: int | None = 512,
    slice_width: int | None = 512,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
) -> tuple[dict, str]:
    """Slice large images given in a directory into smaller windows.

    If output_dir is given, export sliced images and coco file.

    Args:
        coco_annotation_file_path (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            dataset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool, optional): If True, images without annotations
            are ignored. Defaults to False.
        slice_height (int, optional): Height of each slice. Default 512.
        slice_width (int, optional): Width of each slice. Default 512.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image.

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """
    # read coco file
    coco_dict: dict = load_json(coco_annotation_file_path)  # type: ignore[assignment]
    # create image_id_to_annotation_list mapping
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images: list = []

    # iterate over images and slice
    for idx, coco_image in enumerate(tqdm(coco.images)):
        # get image path
        image_path: str = os.path.join(image_dir, coco_image.file_name)
        # get annotation json list corresponding to selected coco image
        # slice image
        try:
            slice_image_result = slice_image(
                image=image_path,
                coco_annotation_list=coco_image.annotations,
                output_file_name=f"{Path(coco_image.file_name).stem}_{idx}",
                output_dir=output_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_area_ratio=min_area_ratio,
                out_ext=out_ext,
                verbose=verbose,
                exif_fix=exif_fix,
            )
            # append slice outputs
            sliced_coco_images.extend(slice_image_result.coco_images)
        except TopologicalError:
            logger.warning(f"Invalid annotation found, skipping this image: {image_path}")

    # create and save coco dict
    ignore_negative_samples_val: bool = ignore_negative_samples if ignore_negative_samples is not None else False
    sliced_coco_dict = create_coco_dict(
        sliced_coco_images, coco_dict["categories"], ignore_negative_samples=ignore_negative_samples_val
    )
    save_path: str = ""
    if output_coco_annotation_file_name and output_dir:
        save_path = str(Path(output_dir) / (output_coco_annotation_file_name + "_coco.json"))
        save_json(sliced_coco_dict, save_path)

    return sliced_coco_dict, save_path


def calc_ratio_and_slice(
    orientation: Literal["vertical", "horizontal", "square"], slide: int = 1, ratio: float = 0.1
) -> tuple[int, int, float, float]:
    """Calculate overlap params according to image resolution.

    Args:
        orientation: image capture angle.
        slide: sliding window.
        ratio: buffer value.

    Returns:
        overlap params.
    """
    if orientation == "vertical":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide * 2, ratio, ratio
    elif orientation == "horizontal":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide * 2, slide, ratio, ratio
    elif orientation == "square":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide, ratio, ratio
    else:
        raise ValueError(f"Invalid orientation: {orientation}. Must be one of 'vertical', 'horizontal', or 'square'.")

    return slice_row, slice_col, overlap_height_ratio, overlap_width_ratio


def calc_resolution_factor(resolution: int) -> int:
    """Calculate power(2,n) and return the closest smaller `n` for resolution.

    Args:
        resolution: the width and height of the image multiplied. such as 1024x720 = 737280.

    Returns:
        Power value of 2 closest to the resolution.
    """
    expo = 0
    while np.power(2, expo) < resolution:
        expo += 1

    return expo - 1


def calc_aspect_ratio_orientation(width: int, height: int) -> Literal["vertical", "horizontal", "square"]:
    """Calculate image capture orientation from aspect ratio.

    Args:
        width: image width.
        height: image height.

    Returns:
        image capture orientation.
    """
    if width < height:
        return "vertical"
    elif width > height:
        return "horizontal"
    else:
        return "square"


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: Literal["vertical", "horizontal", "square"]
) -> tuple[int, int, int, int]:
    """Calculate slice and overlap params according to image resolution.

    Args:
        resolution: str
        height: int
        width: int
        orientation: str.

    Returns:
        x_overlap, y_overlap, slice_width, slice_height
    """
    if resolution == "medium":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=1, ratio=0.8
        )

    elif resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=2, ratio=0.4
        )

    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=4, ratio=0.4
        )
    else:  # low condition
        split_col = 1
        split_row = 1
        overlap_width_ratio = 1
        overlap_height_ratio = 1

    slice_height = height // split_col
    slice_width = width // split_row

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)

    return x_overlap, y_overlap, slice_width, slice_height


def get_resolution_selector(res: str, height: int, width: int) -> tuple[int, int, int, int]:
    """Get slicing parameters based on resolution.

    Args:
        res: resolution of image such as low, medium.
        height: image height.
        width: image width.

    Returns:
        overlap params from slicing params function.
    """
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    x_overlap, y_overlap, slice_width, slice_height = calc_slice_and_overlap_params(
        resolution=res, height=height, width=width, orientation=orientation
    )

    return x_overlap, y_overlap, slice_width, slice_height


def get_auto_slice_params(height: int, width: int) -> tuple[int, int, int, int]:
    """Calculate overlap sliding window and buffer params from image dimensions.

    Factor is the power value of 2 closest to the image resolution:
        - factor <= 18: low resolution image such as 300x300, 640x640
        - 18 < factor <= 21: medium resolution image such as 1024x1024, 1336x960
        - 21 < factor <= 24: high resolution image such as 2048x2048, 2048x4096, 4096x4096
        - factor > 24: ultra-high resolution image such as 6380x6380, 4096x8192.

    Args:
        height: image height.
        width: image width.

    Returns:
        slicing overlap params x_overlap, y_overlap, slice_width, slice_height.
    """
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if factor <= 18:
        return get_resolution_selector("low", height=height, width=width)
    elif 18 <= factor < 21:
        return get_resolution_selector("medium", height=height, width=width)
    elif 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    else:
        return get_resolution_selector("ultra-high", height=height, width=width)


def shift_bboxes(bboxes: Any, offset: Sequence[int]) -> Any:
    """Shift bboxes w.r.t offset.

    Supports Tensor, np.ndarray, and list inputs.

    Args:
        bboxes (Tensor, np.ndarray, list): The bboxes need to be translated. Its shape can
            be (n, 4), which means (x, y, x, y).
        offset (Sequence[int]): The translation offsets with shape of (2, ).

    Returns:
        Tensor, np.ndarray, list: Shifted bboxes.
    """
    shifted_bboxes = []

    if type(bboxes).__module__ == "torch":
        bboxes_is_torch_tensor = True
    else:
        bboxes_is_torch_tensor = False

    # Assert bboxes is iterable
    assert hasattr(bboxes, "__iter__"), "bboxes must be iterable"

    for bbox in bboxes:  # type: ignore[union-attr]
        if bboxes_is_torch_tensor or isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        bbox = BoundingBox(bbox, shift_amount=tuple(offset[:2]))  # type: ignore[arg-type]
        bbox = bbox.get_shifted_box()
        shifted_bboxes.append(bbox.to_xyxy())

    if isinstance(bboxes, np.ndarray):
        return np.stack(shifted_bboxes, axis=0)
    elif bboxes_is_torch_tensor:
        return bboxes.new_tensor(shifted_bboxes)  # type: ignore[attr-defined]
    else:
        return shifted_bboxes


def shift_masks(masks: np.ndarray, offset: Sequence[int], full_shape: Sequence[int]) -> np.ndarray:
    """Shift masks to the original image.

    Args:
        masks (np.ndarray): masks that need to be shifted.
        offset (Sequence[int]): The offset to translate with shape of (2, ).
        full_shape (Sequence[int]): A (height, width) tuple of the huge image's shape.

    Returns:
        np.ndarray: Shifted masks.
    """
    # empty masks
    if masks is None:
        return masks

    shifted_masks = []
    for mask_seg in masks:
        mask = Mask(segmentation=mask_seg, shift_amount=list(offset[:2]), full_shape=list(full_shape[:2]))  # type: ignore[arg-type]
        mask = mask.get_shifted_mask()
        shifted_masks.append(mask.bool_mask)

    return np.stack(shifted_masks, axis=0)
