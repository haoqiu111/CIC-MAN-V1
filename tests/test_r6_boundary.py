from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r8_only_tokens_are_absent_from_code() -> None:
    forbidden = ("v6ic_pair_guard", "lambda_pair_consistency", "external_cwru")
    code = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for base in (ROOT / "src", ROOT / "scripts")
        for path in base.rglob("*.py")
    )
    for token in forbidden:
        assert token not in code
