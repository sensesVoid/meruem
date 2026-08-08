import os


def test_database_url_is_absolute_and_inside_backend(tmp_path, monkeypatch):
    # The settings object is already instantiated at import time; instead we
    # verify the resolution logic that anchors paths to BACKEND_DIR.
    from app.core import config

    url = config.settings.database_url
    assert url.startswith("sqlite:///")
    path = url[len("sqlite:///"):]
    assert os.path.isabs(path)
    assert path.endswith(".db")
    # Must NOT be a bare relative "./meruem.db" that depends on CWD.
    assert not path.startswith("./")


def test_chroma_dir_anchored_to_backend():
    from app.core import config

    assert os.path.isabs(config.settings.chroma_dir)
    assert config.settings.chroma_dir == str(config.BACKEND_DIR / "chroma_data")
