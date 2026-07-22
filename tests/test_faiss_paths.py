from pathlib import Path

from backend.faiss_search import resolve_faiss_paths


def test_resolve_faiss_paths_uses_dataset_specific_assets(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "scifact_index.index").write_bytes(b"index")
    (models_dir / "scifact_metadata.pkl").write_bytes(b"metadata")

    index_path, metadata_path = resolve_faiss_paths("scifact", base_dir=tmp_path)

    assert index_path == str(models_dir / "scifact_index.index")
    assert metadata_path == str(models_dir / "scifact_metadata.pkl")
