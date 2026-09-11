from pathlib import Path

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
