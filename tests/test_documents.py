from pathlib import Path

import pytest
from PIL import Image

from table_qa_agent.config import DocumentConfig
from table_qa_agent.documents import DocumentProcessor
from table_qa_agent.schemas import RegionRef


def test_image_is_normalized_and_cached(tmp_path: Path) -> None:
    source = tmp_path / "table.png"
    Image.new("RGBA", (800, 400), (255, 255, 255, 128)).save(source)
    processor = DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache", max_long_edge=512))

    first = processor.prepare(source)
    second = processor.prepare(source)

    assert first == second
    assert len(first) == 1
    with Image.open(first[0]) as image:
        assert image.mode == "RGB"
        assert max(image.size) == 512
    assert processor.to_data_url(first[0]).startswith("data:image/jpeg;base64,")


def test_high_resolution_region_is_cropped_from_original_image(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    Image.new("RGB", (1000, 500), "white").save(source)
    processor = DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache"))

    outputs = processor.prepare_regions(
        source,
        [RegionRef(page=1, bbox=(0.25, 0.2, 0.75, 0.8))],
        padding_ratio=0,
    )

    with Image.open(outputs[0]) as image:
        assert image.size == (500, 300)


@pytest.mark.parametrize("angle,size", [(0, (100, 60)), (90, (60, 100)),
                                       (180, (100, 60)), (270, (60, 100))])
def test_region_rotation_is_explicit_and_cache_safe(
    tmp_path: Path, angle: int, size: tuple[int, int],
) -> None:
    source = tmp_path / "table.png"
    original = Image.new("RGB", (100, 60), "white")
    original.paste("red", (0, 0, 40, 20))
    original.save(source)
    processor = DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache"))
    ref = RegionRef(page=1, bbox=(0, 0, 1, 1), rotation_degrees=angle)
    output = processor.prepare_regions(source, [ref], padding_ratio=0)
    assert output == processor.prepare_regions(source, [ref], padding_ratio=0)
    with Image.open(output[0]) as result:
        assert result.size == size
        if angle == 90:
            red, green, blue = result.getpixel((50, 10))
            assert red > 180 and green < 80 and blue < 80
    unrotated = processor.prepare_regions(
        source, [RegionRef(page=1, bbox=(0, 0, 1, 1))], padding_ratio=0,
    )
    assert (output == unrotated) == (angle == 0)
    with Image.open(source) as unchanged:
        assert unchanged.size == (100, 60)
