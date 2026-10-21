import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_stubs.py"


def test_stubs_are_up_to_date():
    spec = importlib.util.spec_from_file_location("gen_stubs", SCRIPT)
    gen_stubs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_stubs)
    for name, contents in gen_stubs.render().items():
        path = gen_stubs.PACKAGE / name
        assert (
            path.exists() and path.read_text() == contents
        ), f"{path} is out of date, run `uv run python scripts/gen_stubs.py`"
