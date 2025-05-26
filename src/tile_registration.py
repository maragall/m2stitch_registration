"""Tile registration module for updating stage coordinates without full stitching."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tifffile
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm

from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position
from ._global_optimization import compute_maximum_spanning_tree
from ._stage_model import compute_image_overlap2
from ._stage_model import filter_by_overlap_and_correlation
from ._stage_model import filter_by_repeatability
from ._stage_model import filter_outliers
from ._stage_model import replace_invalid_translations
from ._translation_computation import interpret_translation
from ._translation_computation import multi_peak_max
from ._translation_computation import pcm
from ._typing_utils import BoolArray
from ._typing_utils import Float
from ._typing_utils import NumArray


@dataclass
class ElipticEnvelopPredictor:
    """Elliptic envelope predictor for outlier detection."""
    contamination: float
    epsilon: float
    random_seed: int

    def __call__(self, X: NumArray) -> BoolArray:
        if len(X) < 2:
            return np.ones(len(X), dtype=bool)
        ee = EllipticEnvelope(contamination=self.contamination)
        rng = np.random.default_rng(self.random_seed)
        X = rng.normal(size=X.shape) * self.epsilon + X
        return ee.fit_predict(X) > 0


def read_tiff_images(
    directory: Union[str, Path], pattern: Optional[str] = "*.tiff"
) -> Dict[str, NumArray]:
    """Read all TIFF images from a directory.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF files
    pattern : Optional[str]
        Glob pattern for TIFF files, default "*.tiff"
        
    Returns
    -------
    images : Dict[str, NumArray]
        Dictionary mapping filename to image array
    """
    directory = Path(directory)
    images = {}
    
    # Find all TIFF files
    tiff_files = list(directory.glob(pattern)) + list(directory.glob("*.tif"))
    
    for tiff_file in sorted(tiff_files):
        try:
            image = tifffile.imread(tiff_file)
            images[tiff_file.name] = image
        except Exception as e:
            print(f"Warning: Could not read {tiff_file}: {e}")
            
    return images


def read_coordinates_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read coordinates CSV file.
    
    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to coordinates.csv file
        
    Returns
    -------
    coords : pd.DataFrame
        DataFrame with columns including 'x (mm)', 'y (mm)', 'region', 'fov'
    """
    return pd.read_csv(csv_path)


def extract_tile_indices(
    filenames: List[str], coords_df: pd.DataFrame
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Extract row/col indices from filenames and match with coordinates.
    
    For irregular grids, this creates a spatial mapping where neighbors
    are found by proximity rather than assuming a perfect grid.
    
    Parameters
    ----------
    filenames : List[str]
        List of image filenames
    coords_df : pd.DataFrame
        Coordinates dataframe
        
    Returns
    -------
    rows : List[int]
        Row indices for each image
    cols : List[int]
        Column indices for each image
    filename_to_index : Dict[str, int]
        Mapping from filename to DataFrame index
    """
    # Extract FOV numbers from filenames (assuming format like "manual_X_0_...")
    fov_numbers = []
    for filename in filenames:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:  # First number after 'manual'
                fov_numbers.append(int(part))
                break
    
    # Create mapping from filename to coordinate data
    filename_to_index = {}
    coords_list = []
    
    for idx, (filename, fov) in enumerate(zip(filenames, fov_numbers)):
        # Find matching coordinate entry
        coord_row = coords_df[coords_df['fov'] == fov].iloc[0]
        filename_to_index[filename] = coord_row.name
        coords_list.append((coord_row['x (mm)'], coord_row['y (mm)'], idx))
    
    # Sort by coordinates to create a reasonable row/col assignment
    # Sort primarily by y (for rows), then by x (for columns)
    coords_list.sort(key=lambda x: (x[1], x[0]))
    
    # Assign row/col indices based on sorted order and spatial clustering
    rows = []
    cols = []
    
    # Group by similar y-coordinates (rows)
    y_coords = [c[1] for c in coords_list]
    y_threshold = np.std(y_coords) * 0.1  # Small threshold for grouping
    
    current_row = 0
    last_y = None
    row_assignments = {}
    
    for x, y, original_idx in coords_list:
        if last_y is None or abs(y - last_y) > y_threshold:
            current_row += 1
            last_y = y
        row_assignments[original_idx] = current_row - 1
    
    # Within each row, assign column indices
    rows_dict = {}
    for x, y, original_idx in coords_list:
        row_idx = row_assignments[original_idx]
        if row_idx not in rows_dict:
            rows_dict[row_idx] = []
        rows_dict[row_idx].append((x, original_idx))
    
    col_assignments = {}
    for row_idx, row_tiles in rows_dict.items():
        # Sort by x within each row
        row_tiles.sort(key=lambda x: x[0])
        for col_idx, (x, original_idx) in enumerate(row_tiles):
            col_assignments[original_idx] = col_idx
    
    # Create final row/col lists in original filename order
    for i in range(len(filenames)):
        rows.append(row_assignments[i])
        cols.append(col_assignments[i])
        
    return rows, cols, filename_to_index


def register_tiles(
    images: NumArray,
    rows: List[int],
    cols: List[int],
    overlap_diff_threshold: Float = 10,
    pou: Float = 3,
    ncc_threshold: Float = 0.5,
) -> Tuple[pd.DataFrame, dict]:
    """Register tiles without full stitching - adapted from stitch_images.
    
    Parameters
    ----------
    images : NumArray
        Array of images to register
    rows : List[int]
        Row indices for each image
    cols : List[int]
        Column indices for each image
    overlap_diff_threshold : Float
        Allowed difference from initial guess (percentage)
    pou : Float
        Percent overlap uncertainty
    ncc_threshold : Float
        Normalized cross correlation threshold
        
    Returns
    -------
    grid : pd.DataFrame
        Registration results with pixel positions
    prop_dict : dict
        Dictionary of estimated parameters
    """
    images = np.array(images)
    assert len(rows) == len(cols) == images.shape[0]
    
    sizeY, sizeX = images.shape[1:]
    
    # Create grid DataFrame
    grid = pd.DataFrame({
        "col": cols,
        "row": rows,
    }, index=np.arange(len(cols)))
    
    def get_index(col, row):
        df = grid[(grid["col"] == col) & (grid["row"] == row)]
        return df.index[0] if len(df) == 1 else None
    
    # Find neighbors - more robust for sparse grids
    grid["top"] = grid.apply(
        lambda g: get_index(g["col"], g["row"] - 1), axis=1
    ).astype(pd.Int32Dtype())
    grid["left"] = grid.apply(
        lambda g: get_index(g["col"] - 1, g["row"]), axis=1
    ).astype(pd.Int32Dtype())
    
    # Initialize translation columns to handle cases with no neighbors
    for direction in ["left", "top"]:
        for key in ["ncc", "y", "x"]:
            grid[f"{direction}_{key}_first"] = np.nan
    
    # Translation computation
    for direction in ["left", "top"]:
        for i2, g in tqdm(grid.iterrows(), total=len(grid), desc=f"Computing {direction} translations"):
            i1 = g[direction]
            if pd.isna(i1):
                continue
                
            image1 = images[i1]
            image2 = images[i2]
            
            PCM = pcm(image1, image2).real
            lims = np.array([[-sizeY, sizeY], [-sizeX, sizeX]])
            
            yins, xins, _ = multi_peak_max(PCM)
            max_peak = interpret_translation(
                image1, image2, yins, xins, *lims[0], *lims[1]
            )
            
            for j, key in enumerate(["ncc", "y", "x"]):
                grid.loc[i2, f"{direction}_{key}_first"] = max_peak[j]
    
    # Check if we have valid pairs - handle sparse grids gracefully
    print("\nDebug: Grid structure:")
    print(f"Total tiles: {len(grid)}")
    print(f"Unique rows: {sorted(grid['row'].unique())}")
    print(f"Unique cols: {sorted(grid['col'].unique())}")
    print("\nDebug: Neighbors and NCC values:")
    for i, row in grid.iterrows():
        left_ncc = row.get('left_ncc_first', np.nan)
        top_ncc = row.get('top_ncc_first', np.nan)
        left_ncc_str = f"{left_ncc:.3f}" if not pd.isna(left_ncc) else "NaN"
        top_ncc_str = f"{top_ncc:.3f}" if not pd.isna(top_ncc) else "NaN"
        print(f"  Tile {i}: row={row['row']}, col={row['col']}, left_neighbor={row['left']}, left_ncc={left_ncc_str}, top_neighbor={row['top']}, top_ncc={top_ncc_str}")
    
    has_top_pairs = np.any(grid["top_ncc_first"] > ncc_threshold)
    has_left_pairs = np.any(grid["left_ncc_first"] > ncc_threshold)
    
    print(f"\nDebug: has_left_pairs={has_left_pairs}, has_top_pairs={has_top_pairs}, threshold={ncc_threshold}")
    
    if not has_left_pairs:
        raise ValueError("No good left pairs found - tiles may not have sufficient horizontal overlap")
    
    # For sparse grids, we can proceed with only left pairs if no top pairs exist
    if not has_top_pairs:
        print("Warning: No top pairs found - assuming single-row or sparse grid")
        # Create dummy top displacement for single-row grids
        top_displacement = (0.0, 0.0)
        overlap_top = 50.0  # Default overlap for missing direction
    else:
        # Compute overlaps normally
        predictor = ElipticEnvelopPredictor(contamination=0.4, epsilon=0.01, random_seed=0)
        top_displacement = compute_image_overlap2(
            grid[grid["top_ncc_first"] > ncc_threshold], "top", sizeY, sizeX, predictor
        )
        overlap_top = np.clip(100 - top_displacement[0] * 100, pou, 100 - pou)
    
    # Always compute left displacement (we know we have left pairs)
    predictor = ElipticEnvelopPredictor(contamination=0.4, epsilon=0.01, random_seed=0)
    left_displacement = compute_image_overlap2(
        grid[grid["left_ncc_first"] > ncc_threshold], "left", sizeY, sizeX, predictor
    )
    overlap_left = np.clip(100 - left_displacement[1] * 100, pou, 100 - pou)
    
    # Filter and validate translations
    if has_top_pairs:
        grid["top_valid1"] = filter_by_overlap_and_correlation(
            grid["top_y_first"], grid["top_ncc_first"], overlap_top, sizeY, pou, ncc_threshold
        )
        grid["top_valid2"] = filter_outliers(grid["top_y_first"], grid["top_valid1"])
    else:
        # No top pairs - set all to False
        grid["top_valid1"] = False
        grid["top_valid2"] = False
    
    grid["left_valid1"] = filter_by_overlap_and_correlation(
        grid["left_x_first"], grid["left_ncc_first"], overlap_left, sizeX, pou, ncc_threshold
    )
    grid["left_valid2"] = filter_outliers(grid["left_x_first"], grid["left_valid1"])
    
    # Compute repeatability
    rs = []
    for direction, dims, rowcol in zip(["top", "left"], ["yx", "xy"], ["col", "row"]):
        valid_key = f"{direction}_valid2"
        valid_grid = grid[grid[valid_key]]
        if len(valid_grid) > 0:
            w1s = valid_grid[f"{direction}_{dims[0]}_first"]
            r1 = np.ceil((w1s.max() - w1s.min()) / 2)
            _, w2s = zip(*valid_grid.groupby(rowcol)[f"{direction}_{dims[1]}_first"])
            r2 = np.ceil(np.max([np.max(w2) - np.min(w2) for w2 in w2s]) / 2)
            rs.append(max(r1, r2))
        else:
            rs.append(0)
    r = np.max(rs)
    
    # Apply filters and refinements
    grid = filter_by_repeatability(grid, r, ncc_threshold)
    grid = replace_invalid_translations(grid)
    grid = refine_translations(images, grid, r)
    
    # Compute final positions
    tree = compute_maximum_spanning_tree(grid)
    grid = compute_final_position(grid, tree)
    
    prop_dict = {
        "W": sizeY,
        "H": sizeX,
        "overlap_left": overlap_left,
        "overlap_top": overlap_top,
        "repeatability": r,
    }
    
    return grid, prop_dict


def calculate_pixel_size_microns(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str]
) -> float:
    """Calculate pixel size in microns by comparing stage and pixel positions.
    
    Parameters
    ----------
    grid : pd.DataFrame
        Registration results with pixel positions
    coords_df : pd.DataFrame
        Original coordinates with stage positions in mm
    filename_to_index : Dict[str, int]
        Mapping from filename to coordinate index
    filenames : List[str]
        Ordered list of filenames
        
    Returns
    -------
    pixel_size_um : float
        Pixel size in microns
    """
    # Get pairs of tiles with known displacements
    pixel_distances = []
    stage_distances = []
    
    for idx, row in grid.iterrows():
        # Check left neighbor
        if not pd.isna(row['left']):
            left_idx = int(row['left'])
            
            # Get stage positions for both tiles
            curr_coord_idx = filename_to_index[filenames[idx]]
            left_coord_idx = filename_to_index[filenames[left_idx]]
            
            curr_x_mm = coords_df.loc[curr_coord_idx, 'x (mm)']
            curr_y_mm = coords_df.loc[curr_coord_idx, 'y (mm)']
            left_x_mm = coords_df.loc[left_coord_idx, 'x (mm)']
            left_y_mm = coords_df.loc[left_coord_idx, 'y (mm)']
            
            # Calculate stage distance in microns
            stage_dist_um = np.sqrt(
                ((curr_x_mm - left_x_mm) * 1000) ** 2 +
                ((curr_y_mm - left_y_mm) * 1000) ** 2
            )
            
            # Calculate pixel distance
            curr_x_px = row['x_pos']
            curr_y_px = row['y_pos']
            left_x_px = grid.loc[left_idx, 'x_pos']
            left_y_px = grid.loc[left_idx, 'y_pos']
            
            pixel_dist = np.sqrt(
                (curr_x_px - left_x_px) ** 2 +
                (curr_y_px - left_y_px) ** 2
            )
            
            if pixel_dist > 0:  # Avoid division by zero
                pixel_distances.append(pixel_dist)
                stage_distances.append(stage_dist_um)
    
    # Calculate median pixel size
    if pixel_distances:
        pixel_sizes = np.array(stage_distances) / np.array(pixel_distances)
        pixel_size_um = np.median(pixel_sizes)
        print(f"Calculated pixel size: {pixel_size_um:.4f} µm/pixel")
        return pixel_size_um
    else:
        raise ValueError("Could not calculate pixel size - no valid tile pairs found")


def update_stage_coordinates(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    reference_idx: int = 0
) -> pd.DataFrame:
    """Update stage coordinates based on registration results.
    
    Parameters
    ----------
    grid : pd.DataFrame
        Registration results with pixel positions
    coords_df : pd.DataFrame
        Original coordinates to update
    filename_to_index : Dict[str, int]
        Mapping from filename to coordinate index
    filenames : List[str]
        Ordered list of filenames
    pixel_size_um : float
        Pixel size in microns
    reference_idx : int
        Index of reference tile (default 0)
        
    Returns
    -------
    updated_coords : pd.DataFrame
        Updated coordinates with new stage positions
    """
    updated_coords = coords_df.copy()
    
    # Get reference position
    ref_coord_idx = filename_to_index[filenames[reference_idx]]
    ref_x_mm = coords_df.loc[ref_coord_idx, 'x (mm)']
    ref_y_mm = coords_df.loc[ref_coord_idx, 'y (mm)']
    ref_x_px = grid.loc[reference_idx, 'x_pos']
    ref_y_px = grid.loc[reference_idx, 'y_pos']
    
    # Update all positions
    for idx, row in grid.iterrows():
        coord_idx = filename_to_index[filenames[idx]]
        
        # Calculate position relative to reference in pixels
        delta_x_px = row['x_pos'] - ref_x_px
        delta_y_px = row['y_pos'] - ref_y_px
        
        # Convert to mm
        delta_x_mm = (delta_x_px * pixel_size_um) / 1000.0
        delta_y_mm = (delta_y_px * pixel_size_um) / 1000.0
        
        # Update stage coordinates
        updated_coords.loc[coord_idx, 'x (mm)'] = ref_x_mm + delta_x_mm
        updated_coords.loc[coord_idx, 'y (mm)'] = ref_y_mm + delta_y_mm
        
        # Add registration info
        updated_coords.loc[coord_idx, 'x_pos_px'] = row['x_pos']
        updated_coords.loc[coord_idx, 'y_pos_px'] = row['y_pos']
    
    return updated_coords


def register_and_update_coordinates(
    image_directory: Union[str, Path],
    csv_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    channel_pattern: Optional[str] = None,
    overlap_diff_threshold: Float = 10,
    pou: Float = 3,
    ncc_threshold: Float = 0.5,
) -> pd.DataFrame:
    """Main function to register tiles and update stage coordinates.
    
    Parameters
    ----------
    image_directory : Union[str, Path]
        Directory containing TIFF images
    csv_path : Union[str, Path]
        Path to coordinates.csv
    output_csv_path : Optional[Union[str, Path]]
        Path for output CSV (default: overwrites input)
    channel_pattern : Optional[str]
        Pattern to select specific channel (e.g., "*405_nm_Ex.tiff")
    overlap_diff_threshold : Float
        Allowed difference from initial guess (percentage)
    pou : Float
        Percent overlap uncertainty
    ncc_threshold : Float
        Normalized cross correlation threshold
        
    Returns
    -------
    updated_coords : pd.DataFrame
        Updated coordinates DataFrame
    """
    print("Reading coordinates CSV...")
    coords_df = read_coordinates_csv(csv_path)
    
    print("Reading TIFF images...")
    pattern = channel_pattern or "*.tiff"
    images_dict = read_tiff_images(image_directory, pattern)
    
    if not images_dict:
        raise ValueError(f"No TIFF images found in {image_directory}")
    
    print(f"Found {len(images_dict)} images")
    
    # Sort filenames for consistent ordering
    filenames = sorted(images_dict.keys())
    images = np.array([images_dict[fn] for fn in filenames])
    
    print("Extracting tile indices...")
    rows, cols, filename_to_index = extract_tile_indices(filenames, coords_df)
    
    print("Registering tiles...")
    grid, prop_dict = register_tiles(
        images, rows, cols, 
        overlap_diff_threshold=overlap_diff_threshold,
        pou=pou,
        ncc_threshold=ncc_threshold
    )
    
    print("Calculating pixel size...")
    pixel_size_um = calculate_pixel_size_microns(
        grid, coords_df, filename_to_index, filenames
    )
    
    print("Updating stage coordinates...")
    updated_coords = update_stage_coordinates(
        grid, coords_df, filename_to_index, filenames, pixel_size_um
    )
    
    # Save results
    if output_csv_path is None:
        output_csv_path = csv_path
        
    print(f"Saving updated coordinates to {output_csv_path}...")
    updated_coords.to_csv(output_csv_path, index=False)
    
    # Print summary
    print("\nRegistration complete!")
    print(f"Pixel size: {pixel_size_um:.4f} µm/pixel")
    print(f"Image size: {prop_dict['W']} x {prop_dict['H']} pixels")
    print(f"Overlap (top): {prop_dict['overlap_top']:.1f}%")
    print(f"Overlap (left): {prop_dict['overlap_left']:.1f}%")
    
    return updated_coords 