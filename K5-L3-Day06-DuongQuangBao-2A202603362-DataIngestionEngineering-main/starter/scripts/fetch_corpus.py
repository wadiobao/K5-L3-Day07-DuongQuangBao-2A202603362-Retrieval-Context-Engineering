"""Download the public corpus declared in the repository's data/manifest.json."""
import argparse
import json
from pathlib import Path

from src.ingestion import download_gutenberg_text


STARTER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()

    manifest_path = REPO_ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = REPO_ROOT / "data" / manifest["source_path"]
    if destination.exists() and not args.force:
        print(f"Corpus already present: {destination}")
        return

    path = download_gutenberg_text(manifest["source_url"], destination)
    print(f"Downloaded {manifest['document_id']} to {path}")


if __name__ == "__main__":
    main()
