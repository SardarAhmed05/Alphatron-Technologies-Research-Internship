import os
import pytest
from src.loaders import DocumentIngestor


@pytest.fixture
def sample_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


def test_load_txt_file(sample_dir):
    txt_file = os.path.join(sample_dir, "sample_notes.txt")
    if not os.path.exists(txt_file):
        pytest.skip("sample_notes.txt not found")

    ingestor = DocumentIngestor()
    docs = ingestor.load_file(txt_file)
    assert len(docs) > 0
    assert "Alphatron" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "TXT"


def test_load_csv_file(sample_dir):
    csv_file = os.path.join(sample_dir, "sample_data.csv")
    if not os.path.exists(csv_file):
        pytest.skip("sample_data.csv not found")

    ingestor = DocumentIngestor()
    docs = ingestor.load_file(csv_file)
    assert len(docs) > 0
    assert "Alice Smith" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "Excel/CSV"


def test_unsupported_format_raises_error(tmp_path):
    ingestor = DocumentIngestor()
    # Create an existing file with unsupported extension .xyz
    unsupported_file = tmp_path / "test_file.xyz"
    unsupported_file.write_text("dummy content")

    with pytest.raises(ValueError):
        ingestor.load_file(str(unsupported_file))


def test_nonexistent_file_raises_error():
    ingestor = DocumentIngestor()
    with pytest.raises(FileNotFoundError):
        ingestor.load_file("nonexistent_file.txt")
