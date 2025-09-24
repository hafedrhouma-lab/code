import pathlib
from glob import glob
from pprint import pformat

import click
import pandas as pd


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pattern", default="tests/**/*.parquet", type=click.STRING)
@click.argument("path", default=".", type=click.Path(exists=True))
@click.argument("rows_limit", default=10, type=click.INT)
@click.option("--dryrun/--no-dryrun", default=False, help="Do not update files")
def main(pattern: str, path: click.Path, rows_limit: int, dryrun: bool):
    """
    ./scripts/cut_parquet_files.py --no-dryrun "tests/**/streamlit_*.parquet" "./" 10
    """
    files_pattern = f"{path}/{pattern}"
    click.echo(f"Iteration over all file matching to `{files_pattern}`")

    paths = [pathlib.Path(file_path) for file_path in glob(files_pattern, recursive=True)]
    click.echo(f"Detected files: \n{pformat([str(p) for p in paths])}")

    if dryrun:
        return

    for path in paths:
        df: pd.DataFrame = pd.read_parquet(path)
        limited_df = df.head(rows_limit)
        path.unlink()
        limited_df.to_parquet(path)
        click.echo(
            f"Cut file from {df.shape[0]} to {limited_df.shape[0]} rows, "
            f"new size={path.stat().st_size / (1024 * 1024):.5f} mb: {path}"
        )


if __name__ == "__main__":
    main()
