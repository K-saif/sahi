"""Tests for slicing an image that is read region by region instead of decoded whole."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sahi.slicing import SlicedImage, slice_image
from sahi.utils.cv import read_image, read_image_as_pil, read_image_size
from sahi.utils.lazy_image import LazyImageSource, is_lazy_image_source

IMAGE_PATH = "tests/data/small-vehicles1.jpeg"


class CountingArraySource(LazyImageSource):
    """A lazy source over an in-memory array that records how it is read.

    Stands in for a slide reader so the tests can assert the access pattern -- how many
    regions are read, and how many are alive at once -- without a slide library.
    """

    def __init__(self, array: np.ndarray) -> None:
        """Wrap an HWC array."""
        self.array = array
        self.read_count = 0
        self.peak_alive = 0
        # numpy arrays are not hashable, so alive regions are tracked as plain refs
        self._alive: list[weakref.ref] = []

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the wrapped array's shape."""
        return self.array.shape

    def read_region(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Return a copy of the region, as a real reader would."""
        self.read_count += 1
        # a copy, because a real reader decodes rather than returning a view
        region = np.array(self.array[top : top + height, left : left + width])
        self._alive = [ref for ref in self._alive if ref() is not None]
        self._alive.append(weakref.ref(region))
        self.peak_alive = max(self.peak_alive, len(self._alive))
        return region


@pytest.fixture
def image_array() -> np.ndarray:
    """The test image as an HWC array."""
    return read_image(IMAGE_PATH)


@pytest.fixture(scope="module")
def detection_model():  # noqa: ANN201
    """A loaded YOLO11n model on CPU."""
    from sahi.models.ultralytics import UltralyticsDetectionModel

    from .utils.ultralytics import UltralyticsConstants, download_yolo11n_model

    download_yolo11n_model()
    model = UltralyticsDetectionModel(
        model_path=UltralyticsConstants.YOLO11N_MODEL_PATH,
        confidence_threshold=0.5,
        device="cpu",
        category_remapping=None,
        load_at_init=False,
        image_size=320,
    )
    model.load_model()
    return model


def test_is_lazy_image_source(image_array: np.ndarray) -> None:
    """Only a LazyImageSource is treated as lazy."""
    assert is_lazy_image_source(CountingArraySource(image_array))
    assert not is_lazy_image_source(image_array)
    assert not is_lazy_image_source(IMAGE_PATH)


def test_indexing_matches_the_array_it_stands_in_for(image_array: np.ndarray) -> None:
    """`source[top:bottom, left:right]` returns what the array would."""
    source = CountingArraySource(image_array)
    assert np.array_equal(source[10:150, 20:200], image_array[10:150, 20:200])
    # a bound past the edge clips, as numpy does
    assert np.array_equal(source[500:9999, 0:50], image_array[500:9999, 0:50])
    assert source.ndim == 3


def test_indexing_rejects_what_it_cannot_serve(image_array: np.ndarray) -> None:
    """Unsupported indexing fails loudly instead of returning something wrong."""
    source = CountingArraySource(image_array)
    with pytest.raises(TypeError):
        _ = source[5]
    with pytest.raises(TypeError):
        _ = source[0:10:2, 0:10]
    with pytest.raises(TypeError):
        _ = source[0:10, 0:10, 0]


def test_slice_image_reads_nothing_up_front(image_array: np.ndarray) -> None:
    """Slicing a lazy source plans the slices without reading any pixels."""
    source = CountingArraySource(image_array)
    result = slice_image(image=source, slice_height=256, slice_width=256)

    assert len(result) > 1
    assert source.read_count == 0
    # nothing decoded means nothing to hold on to
    assert result.original_image is None


def test_lazy_slices_match_eager_slices(image_array: np.ndarray) -> None:
    """A lazy source yields exactly the slices the in-memory path yields."""
    kwargs: dict = dict(slice_height=256, slice_width=256, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    eager = slice_image(image=image_array, **kwargs)
    lazy = slice_image(image=CountingArraySource(image_array), **kwargs)

    assert len(eager) == len(lazy)
    assert eager.starting_pixels == lazy.starting_pixels
    for i in range(len(eager)):
        assert np.array_equal(eager.slice_at(i), lazy.slice_at(i))


def test_each_access_reads_exactly_one_region(image_array: np.ndarray) -> None:
    """Reading a slice costs one region read, and no slice is read until asked for."""
    source = CountingArraySource(image_array)
    result = slice_image(image=source, slice_height=256, slice_width=256)

    result.slice_at(0)
    assert source.read_count == 1
    result.slice_at(1)
    assert source.read_count == 2


def test_in_memory_slices_are_still_views(image_array: np.ndarray) -> None:
    """The in-memory path keeps handing out free views, as it did before."""
    result = slice_image(image=image_array, slice_height=256, slice_width=256)
    assert result.slice_at(0).base is not None
    assert result.original_image is not None


def test_read_image_size_does_not_decode(image_array: np.ndarray) -> None:
    """A lazy source can be sized without reading any of it."""
    source = CountingArraySource(image_array)
    height, width = image_array.shape[:2]
    assert read_image_size(source) == (width, height)
    assert source.read_count == 0


def test_read_image_as_pil_refuses_a_lazy_source(image_array: np.ndarray) -> None:
    """Decoding a lazy source whole is refused with an explanation."""
    source = CountingArraySource(image_array)
    with pytest.raises(TypeError, match="LazyImageSource"):
        read_image_as_pil(source)
    assert source.read_count == 0


def test_export_holds_only_the_slices_being_written(image_array: np.ndarray, tmp_path: Path) -> None:
    """Exporting a lazy source's slices writes them all without holding them all."""
    source = CountingArraySource(image_array)
    result = slice_image(
        image=source,
        output_file_name="slice",
        output_dir=str(tmp_path),
        slice_height=256,
        slice_width=256,
    )

    written = sorted(tmp_path.glob("*.png"))
    assert len(written) == len(result)
    assert source.read_count == len(result)
    # the worker pool bounds how many are alive, rather than the slice count
    assert source.peak_alive < len(result)


class TestSlicedImageBackwardCompatibility:
    """The existing SlicedImage and SliceImageResult surface is unchanged."""

    def test_positional_construction_and_attribute_access(self) -> None:
        """A SlicedImage still takes pixels positionally and exposes them as `.image`."""
        from sahi.utils.coco import CocoImage

        pixels = np.zeros((4, 4, 3), dtype=np.uint8)
        sliced = SlicedImage(pixels, CocoImage(file_name="a.png", height=4, width=4), [0, 0])
        assert np.array_equal(sliced.image, pixels)
        assert sliced.starting_pixel == [0, 0]

    def test_image_is_still_assignable(self) -> None:
        """Downstream code that overwrites `.image` keeps working."""
        from sahi.utils.coco import CocoImage

        sliced = SlicedImage(np.zeros((4, 4, 3), np.uint8), CocoImage(file_name="a.png", height=4, width=4), [0, 0])
        replacement = np.ones((4, 4, 3), np.uint8)
        sliced.image = replacement
        assert np.array_equal(sliced.image, replacement)

    def test_result_collection_api(self, image_array: np.ndarray) -> None:
        """The list properties and item access behave as before."""
        result = slice_image(image=image_array, slice_height=256, slice_width=256)

        assert len(result.images) == len(result)
        assert len(result.starting_pixels) == len(result)
        assert len(result.coco_images) == len(result)
        assert len(result.filenames) == len(result)
        assert len(result.sliced_image_list) == len(result)

        first = result[0]
        assert isinstance(first, dict)
        assert set(first) == {"image", "coco_image", "starting_pixel", "filename"}
        assert np.array_equal(first["image"], result.slice_at(0))
        assert isinstance(result[0:2], list)

    def test_missing_pixels_and_source_is_an_error(self) -> None:
        """A SlicedImage with neither pixels nor a source says so."""
        from sahi.utils.coco import CocoImage

        sliced = SlicedImage(None, CocoImage(file_name="a.png", height=4, width=4), [0, 0])
        with pytest.raises(ValueError, match="neither pixels nor a source"):
            _ = sliced.image


class TestSlicedPredictionOnALazySource:
    """Sliced prediction over a lazy source matches the in-memory run and stays bounded."""

    SLICE_SIZE = 256
    BATCH_SIZE = 4

    def _predict(self, image, detection_model, slice_size: int | None = None, **kwargs: Any):  # noqa: ANN001, ANN202
        from sahi.predict import get_sliced_prediction

        slice_size = slice_size or self.SLICE_SIZE
        return get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            batch_size=self.BATCH_SIZE,
            verbose=0,
            **kwargs,
        )

    @staticmethod
    def _boxes(result) -> list[list[float]]:  # noqa: ANN001
        """Detections as sorted bbox corners, for comparison."""
        return sorted(prediction.bbox.to_xyxy() for prediction in result.object_prediction_list)

    def test_detections_match_the_in_memory_run(self, image_array: np.ndarray, detection_model) -> None:  # noqa: ANN001
        """The same image sliced lazily gives the same detections."""
        eager = self._predict(image_array, detection_model, perform_standard_pred=False)
        lazy = self._predict(CountingArraySource(image_array), detection_model, perform_standard_pred=False)

        assert len(lazy.object_prediction_list) > 0
        assert self._boxes(lazy) == self._boxes(eager)

    def test_slices_held_at_once_do_not_grow_with_slice_count(
        self,
        image_array: np.ndarray,
        detection_model,  # noqa: ANN001
    ) -> None:
        """What is held at once tracks the batch size, not how many slices there are.

        This is the property that lets an image larger than memory be predicted on. The
        bound is two batches rather than one: a model keeps the batch it was last given
        (`_original_predictions`, or `_batch_images` on the base class) until it is handed
        the next one, which happens after that next batch has been read.
        """
        coarse = CountingArraySource(image_array)
        self._predict(coarse, detection_model, slice_size=256, perform_standard_pred=False)

        fine = CountingArraySource(image_array)
        result = self._predict(fine, detection_model, slice_size=128, perform_standard_pred=False)

        assert fine.read_count > coarse.read_count * 2, "the finer run must produce many more slices"
        assert coarse.read_count > self.BATCH_SIZE * 2, "need enough slices for the bound to mean something"

        # four times the slices, no more memory
        assert fine.peak_alive == coarse.peak_alive
        assert fine.peak_alive <= 2 * self.BATCH_SIZE
        assert fine.peak_alive < fine.read_count
        assert len(result.object_prediction_list) > 0

    def test_standard_pred_is_skipped_rather_than_decoding_the_source(
        self,
        image_array: np.ndarray,
        detection_model,  # noqa: ANN001
    ) -> None:
        """The default full-image pass is dropped instead of decoding the whole source."""
        source = CountingArraySource(image_array)
        # perform_standard_pred defaults to True, which would need the whole image at once
        with_default = self._predict(source, detection_model)
        without = self._predict(CountingArraySource(image_array), detection_model, perform_standard_pred=False)

        assert self._boxes(with_default) == self._boxes(without)
        # every read is a slice, so nothing read the source as one whole image
        assert source.peak_alive <= 2 * self.BATCH_SIZE
        assert source.read_count == len(slice_image(image=image_array, slice_height=256, slice_width=256))

    def test_result_reports_the_source_size(self, image_array: np.ndarray, detection_model) -> None:  # noqa: ANN001
        """The prediction result knows the image size without decoding it."""
        height, width = image_array.shape[:2]
        result = self._predict(CountingArraySource(image_array), detection_model, perform_standard_pred=False)
        assert (result.image_width, result.image_height) == (width, height)


class StubSlide:
    """The part of the OpenSlide reader interface `SlideImageSource` uses."""

    def __init__(self, array: np.ndarray, downsamples: tuple[float, ...] = (1.0,)) -> None:
        """Serve regions of `array` as the pyramid levels described by `downsamples`."""
        self.array = array
        self.level_downsamples = downsamples
        height, width = array.shape[:2]
        self.level_dimensions = [(int(width // d), int(height // d)) for d in downsamples]
        self.requested: list[tuple] = []
        self.closed = False

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> np.ndarray:
        """Return an RGBA region, as a slide reader does."""
        self.requested.append((location, level, size))
        left, top = location
        width, height = size
        downsample = int(self.level_downsamples[level])
        region = self.array[top::downsample, left::downsample][:height, :width]
        alpha = np.full((*region.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate([region, alpha], axis=2)

    def close(self) -> None:
        """Record that the handle was closed."""
        self.closed = True


class TestSlideImageSource:
    """The OpenSlide-style adapter."""

    def test_reads_regions_and_drops_alpha(self, image_array: np.ndarray) -> None:
        """A region matches the array, without the alpha channel the reader adds."""
        from sahi.utils.lazy_image import SlideImageSource

        source = SlideImageSource(StubSlide(image_array))
        height, width = image_array.shape[:2]

        assert source.shape == (height, width, 3)
        region = source[10:150, 20:200]
        assert region.shape == (140, 180, 3)
        assert np.array_equal(region, image_array[10:150, 20:200])

    def test_a_lower_level_is_addressed_in_level_zero_coordinates(self, image_array: np.ndarray) -> None:
        """`read_region` takes level 0 coordinates whatever level it reads."""
        from sahi.utils.lazy_image import SlideImageSource

        slide = StubSlide(image_array, downsamples=(1.0, 2.0))
        source = SlideImageSource(slide, level=1)

        height, width = image_array.shape[:2]
        assert source.shape == (height // 2, width // 2, 3)

        source[50:100, 30:80]
        location, level, size = slide.requested[-1]
        assert level == 1
        # the level 1 origin (30, 50) is (60, 100) at level 0
        assert location == (60, 100)
        assert size == (50, 50)

    def test_closing_closes_the_handle(self, image_array: np.ndarray) -> None:
        """The context manager closes the slide it was given."""
        from sahi.utils.lazy_image import SlideImageSource

        slide = StubSlide(image_array)
        with SlideImageSource(slide):
            pass
        assert slide.closed

    def test_slices_through_the_pipeline(self, image_array: np.ndarray) -> None:
        """Slicing a slide source gives the slices the in-memory path gives."""
        from sahi.utils.lazy_image import SlideImageSource

        kwargs: dict = dict(slice_height=256, slice_width=256)
        eager = slice_image(image=image_array, **kwargs)
        lazy = slice_image(image=SlideImageSource(StubSlide(image_array)), **kwargs)

        assert len(eager) == len(lazy)
        for i in range(len(eager)):
            assert np.array_equal(eager.slice_at(i), lazy.slice_at(i))


class TestTiffFileImageSource:
    """The tifffile adapter, over a tiled pyramidal TIFF."""

    @pytest.fixture
    def pyramid_path(self, image_array: np.ndarray, tmp_path: Path) -> str:
        """A tiled, pyramidal TIFF holding the test image."""
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("zarr")

        path = tmp_path / "pyramid.tiff"
        with tifffile.TiffWriter(str(path)) as writer:
            writer.write(image_array, tile=(128, 128), photometric="rgb", subifds=1)
            writer.write(image_array[::2, ::2], tile=(128, 128), photometric="rgb", subfiletype=1)
        return str(path)

    def test_reads_regions_of_the_full_resolution_level(self, pyramid_path: str, image_array: np.ndarray) -> None:
        """A region matches the source image."""
        from sahi.utils.lazy_image import TiffFileImageSource

        with TiffFileImageSource(pyramid_path) as source:
            assert source.shape[:2] == image_array.shape[:2]
            assert np.array_equal(source[10:150, 20:200], image_array[10:150, 20:200])

    def test_reads_a_lower_pyramid_level(self, pyramid_path: str, image_array: np.ndarray) -> None:
        """A lower level reports and serves its own smaller shape."""
        from sahi.utils.lazy_image import TiffFileImageSource

        with TiffFileImageSource(pyramid_path, level=1) as source:
            height, width = image_array.shape[:2]
            assert source.shape[:2] == (height // 2, width // 2)
            assert source[0:32, 0:32].shape == (32, 32, 3)

    def test_slices_through_the_pipeline(self, pyramid_path: str, image_array: np.ndarray) -> None:
        """Slicing a TIFF source gives the slices the in-memory path gives."""
        from sahi.utils.lazy_image import TiffFileImageSource

        kwargs: dict = dict(slice_height=256, slice_width=256)
        eager = slice_image(image=image_array, **kwargs)
        with TiffFileImageSource(pyramid_path) as source:
            lazy = slice_image(image=source, **kwargs)
            assert len(eager) == len(lazy)
            for i in range(len(eager)):
                assert np.array_equal(eager.slice_at(i), lazy.slice_at(i))


class TestAutomaticLazyReading:
    """A path to an image too large to decode is read in regions without being asked."""

    @pytest.fixture
    def tiled_tiff(self, image_array: np.ndarray, tmp_path: Path) -> str:
        """A tiled TIFF holding the test image."""
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("zarr")
        path = tmp_path / "tiled.tiff"
        with tifffile.TiffWriter(str(path)) as writer:
            writer.write(image_array, tile=(128, 128), photometric="rgb")
        return str(path)

    @pytest.fixture
    def untiled_tiff(self, image_array: np.ndarray, tmp_path: Path) -> str:
        """A strip-based TIFF, which cannot be read region by region."""
        tifffile = pytest.importorskip("tifffile")
        path = tmp_path / "untiled.tiff"
        tifffile.imwrite(str(path), image_array, photometric="rgb")
        return str(path)

    @pytest.fixture
    def tiny_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Treat any image as too large to decode."""
        import sahi.utils.lazy_image as lazy_image_module

        monkeypatch.setattr(lazy_image_module, "LARGE_IMAGE_THRESHOLD_BYTES", 1)

    def test_ordinary_images_are_untouched(self, image_array: np.ndarray, tmp_path: Path) -> None:
        """Nothing under the threshold is diverted onto the lazy path."""
        from sahi.utils.lazy_image import open_large_image_source

        assert open_large_image_source(IMAGE_PATH) is None
        assert open_large_image_source(image_array) is None
        assert open_large_image_source("https://example.com/scan.tiff") is None
        assert open_large_image_source(tmp_path / "missing.tiff") is None

    def test_a_small_tiff_is_still_decoded_normally(self, tiled_tiff: str) -> None:
        """Being a tiled TIFF is not enough on its own."""
        from sahi.utils.lazy_image import open_large_image_source

        assert open_large_image_source(tiled_tiff) is None

        result = slice_image(image=tiled_tiff, slice_height=256, slice_width=256)
        assert not result.is_lazy
        assert result.original_image is not None

    def test_a_large_tiled_tiff_is_read_in_regions(self, tiled_tiff: str, tiny_threshold: None) -> None:
        """Over the threshold, a tiled TIFF is sliced without being decoded."""
        result = slice_image(image=tiled_tiff, slice_height=256, slice_width=256)

        assert result.is_lazy
        assert result.original_image is None
        result.close()

    def test_a_large_untiled_tiff_is_left_alone(self, untiled_tiff: str, tiny_threshold: None) -> None:
        """An untiled image cannot be read in regions, so it is decoded as before."""
        from sahi.utils.lazy_image import open_large_image_source

        assert open_large_image_source(untiled_tiff) is None

        result = slice_image(image=untiled_tiff, slice_height=256, slice_width=256)
        assert not result.is_lazy

    def test_auto_lazy_can_be_turned_off(self, tiled_tiff: str, tiny_threshold: None) -> None:
        """`auto_lazy=False` always decodes, whatever the size."""
        result = slice_image(image=tiled_tiff, slice_height=256, slice_width=256, auto_lazy=False)
        assert not result.is_lazy

    def test_slices_match_the_decoded_path(
        self,
        tiled_tiff: str,
        image_array: np.ndarray,
        tiny_threshold: None,
    ) -> None:
        """Reading in regions gives the slices decoding would have given."""
        kwargs: dict = dict(slice_height=256, slice_width=256)
        eager = slice_image(image=image_array, auto_lazy=False, **kwargs)
        lazy = slice_image(image=tiled_tiff, **kwargs)

        assert lazy.is_lazy
        assert len(eager) == len(lazy)
        for i in range(len(eager)):
            assert np.array_equal(eager.slice_at(i), lazy.slice_at(i))
        lazy.close()

    def test_a_source_opened_for_us_is_closed_again(self, tiled_tiff: str, tiny_threshold: None) -> None:
        """The result closes the handle it opened, and only that one."""
        result = slice_image(image=tiled_tiff, slice_height=256, slice_width=256)
        assert result._owned_source is not None

        result.close()
        assert result._owned_source is None
        result.close()  # closing twice is not an error

    def test_a_source_the_caller_opened_stays_open(self, image_array: np.ndarray) -> None:
        """Closing a result never closes a source the caller passed in."""
        source = CountingArraySource(image_array)
        result = slice_image(image=source, slice_height=256, slice_width=256)

        assert result._owned_source is None
        result.close()
        # still usable, because it was never this result's to close
        assert result.slice_at(0).shape[:2] == (256, 256)


def test_prediction_from_a_path_alone_matches_the_decoded_run(
    image_array: np.ndarray,
    tmp_path: Path,
    detection_model,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a path to an image too large to decode gives the same detections.

    The point of the whole feature: the caller passes a path, as they always have, and
    the image is read in regions because it has to be.
    """
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("zarr")
    import sahi.utils.lazy_image as lazy_image_module
    from sahi.predict import get_sliced_prediction

    path = tmp_path / "tiled.tiff"
    with tifffile.TiffWriter(str(path)) as writer:
        writer.write(image_array, tile=(128, 128), photometric="rgb")

    def boxes(result):  # noqa: ANN001, ANN202
        return sorted(prediction.bbox.to_xyxy() for prediction in result.object_prediction_list)

    common: dict = dict(
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        perform_standard_pred=False,
        verbose=0,
    )
    from_array = get_sliced_prediction(image=image_array, **common)

    monkeypatch.setattr(lazy_image_module, "LARGE_IMAGE_THRESHOLD_BYTES", 1)
    from_path = get_sliced_prediction(image=str(path), **common)

    assert len(from_array.object_prediction_list) > 0
    assert boxes(from_path) == boxes(from_array)
    assert (from_path.image_width, from_path.image_height) == image_array.shape[:2][::-1]
