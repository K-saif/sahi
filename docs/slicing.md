---
tags:
  - slicing
  - api-reference
  - coco
  - dataset
  - small-object-detection
---

# Slicing

## Slicing

::: sahi.slicing

### Slicing Utilities

- Slice an image:

```python
from sahi.slicing import slice_image

slice_image_result = slice_image(
    image=image_path,
    output_file_name=output_file_name,
    output_dir=output_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

- Slice a COCO formatted dataset:

```python
from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

### Images larger than memory

Slicing normally decodes the whole image once and hands every slice a view into it. A
gigapixel scan cannot be decoded that way, so a tiled TIFF too large to fit is read
region by region instead. Nothing is needed to switch this on:

```python
result = get_sliced_prediction(
    image="scan.tiff",  # 205176 x 49607, 30.5 GB decoded
    detection_model=detection_model,
    slice_height=1024,
    slice_width=1024,
)
```

Only slices in use are held, so memory tracks the batch size rather than the image. This
needs `pip install sahi[wsi]`; without it the image is decoded as before. An image under
`LARGE_IMAGE_THRESHOLD_BYTES` (2 GB decoded) is never diverted, so ordinary use is
unaffected, and `auto_lazy=False` on `slice_image` always decodes.

Untiled images are decoded whole even when they are large, because every region would
have to be decoded from the start of the file. sahi logs a warning when it meets one.

#### Choosing the backend

To read through a slide library, or to work at a lower pyramid level, pass a source
yourself. `SlideImageSource` wraps any handle with the OpenSlide reader interface, so
`openslide`, `tiffslide` and `cucim` all work:

```python
import tiffslide

from sahi.utils.lazy_image import SlideImageSource

with SlideImageSource(tiffslide.TiffSlide("scan.svs"), level=1) as source:
    result = get_sliced_prediction(image=source, detection_model=detection_model)
```

`TiffFileImageSource` is the `tifffile` equivalent, and is what the automatic path uses.
Anything that indexes as `source[top:bottom, left:right]` and reports a `shape` can serve:
subclass `LazyImageSource` and implement `shape` and `read_region` for another backend.

A source sahi opened for you is closed with the result. One you opened stays yours to
close, so the context manager above still matters.

#### What changes

`perform_standard_pred` is skipped when slices are read lazily. That pass predicts on the
whole image at once, which is exactly what cannot be afforded here, so objects larger than
a single slice may be missed. sahi logs when it skips it.

### Interactive Demo

Want to experiment with different slicing parameters and see their effects?
Check out our [interactive notebooks](notebooks.md) for hands-on examples.
