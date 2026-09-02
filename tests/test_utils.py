from src.utils import normalize_text, sha256_text


def test_normalize_text_stable():
    assert normalize_text(" Hello   WORLD\n") == "hello world"
    assert sha256_text("x") == sha256_text("x")
