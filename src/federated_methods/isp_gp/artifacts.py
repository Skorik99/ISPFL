import csv
from pathlib import Path


class ArtifactLogger:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def append(self, filename: str, row: dict) -> None:
        path = self.directory / filename
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
