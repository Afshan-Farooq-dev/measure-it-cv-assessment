"""Publish a complete run to a new directory without replacing existing evidence."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def new_output_directory(destination: Path, sources=()):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}. Choose a new --output directory.")
    for source in sources:
        source = Path(source).resolve()
        if destination == source or source in destination.parents:
            raise ValueError("Output must be outside the source directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        yield staging
        if destination.exists():
            raise FileExistsError(f"Output appeared during this run: {destination}")
        staging.rename(destination)
