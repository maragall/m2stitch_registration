"""Command-line interface for tile registration."""
import click
from pathlib import Path

from .tile_registration import register_and_update_coordinates


@click.command()
@click.argument('image_directory', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('csv_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output-csv', '-o', type=click.Path(), default=None,
              help='Output CSV path (default: overwrites input CSV)')
@click.option('--channel-pattern', '-c', default=None,
              help='Pattern to select specific channel (e.g., "*405_nm_Ex.tiff")')
@click.option('--overlap-diff-threshold', default=10, type=float,
              help='Allowed difference from initial guess in percentage (default: 10)')
@click.option('--pou', default=3, type=float,
              help='Percent overlap uncertainty (default: 3)')
@click.option('--ncc-threshold', default=0.5, type=float,
              help='Normalized cross correlation threshold (default: 0.5)')
@click.version_option()
def main(image_directory, csv_path, output_csv, channel_pattern, 
         overlap_diff_threshold, pou, ncc_threshold):
    """Register microscope image tiles and update stage coordinates.
    
    IMAGE_DIRECTORY: Directory containing TIFF images
    CSV_PATH: Path to coordinates.csv file
    """
    register_and_update_coordinates(
        image_directory=Path(image_directory),
        csv_path=Path(csv_path),
        output_csv_path=Path(output_csv) if output_csv else None,
        channel_pattern=channel_pattern,
        overlap_diff_threshold=overlap_diff_threshold,
        pou=pou,
        ncc_threshold=ncc_threshold
    )


if __name__ == "__main__":
    main(prog_name="m2stitch")  # pragma: no cover
