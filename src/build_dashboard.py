"""Assemble everything already on disk (results/*.json, no training) into one offline dashboard.html."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(pattern):
    out = {}
    for f in RESULTS.glob(pattern):
        out[f.stem] = json.loads(f.read_text())
    return out


def main():
    runs = []
    for f in sorted(RESULTS.glob("*_*.json")):
        if f.stem.startswith(("eda_", "features_")):
            continue
        runs.append(json.loads(f.read_text()))

    eda_raw = load("eda_*.json")
    eda = {k.replace("eda_", ""): v for k, v in eda_raw.items()}

    features_raw = load("features_*.json")
    features = {k.replace("features_", ""): v for k, v in features_raw.items()}

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runs": runs,
        "eda": eda,
        "features": features,
    }

    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data)
    html = template.replace("/*__DATA__*/{}/*__END_DATA__*/", payload)

    out_path = ROOT / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.1f} MB) — {len(runs)} runs, {len(eda)} datasets, {len(features)} feature sets")


if __name__ == "__main__":
    main()
