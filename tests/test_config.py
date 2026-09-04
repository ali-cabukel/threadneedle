from threadneedle.config import PROJECT_ROOT, settings


def test_roots_point_at_the_repo():
    assert (PROJECT_ROOT / "sources.yaml").exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert settings.manifest_path == PROJECT_ROOT / "sources.yaml"
    assert settings.static_dir == PROJECT_ROOT / "static"
    assert settings.pdf_parser == "docling"
    assert settings.docling_do_ocr is False
