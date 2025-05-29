"""Tile registration module for updating stage coordinates without full stitching."""
import os
from dataclasses import dataclass
from pathlib import Path
import re
import logging

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from typing import Dict, List, Optional, Tuple, Union, Pattern
from ._typing_utils import BoolArray
from ._typing_utils import Float
from ._typing_utils import NumArray

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

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class RowInfo:
    """Information about a row of tiles."""
    center_y: float
    tile_indices: List[int]

# Will have to update if acquisition folder structuring and naming changes 
DEFAULT_FOV_RE = re.compile(r"(?:^|_)0*(0|[1-9]\d*)_(?:\d+)_", re.I)  # fov first number followed by _<z>
DEFAULT_FOV_COL = "fov"
DEFAULT_X_COL = "x (mm)"
DEFAULT_Y_COL = "y (mm)"

# Constants for tile registration
MIN_PITCH_FOR_FACTOR = 0.1  # Minimum pitch value to use factor-based tolerance
DEFAULT_ABSOLUTE_TOLERANCE = 0.05  # Default absolute tolerance in mm

def extract_tile_indices(
    filenames: List[str],
    coords_df: pd.DataFrame,
    *,
    fov_re: Pattern[str] = DEFAULT_FOV_RE,
    fov_col_name: str = DEFAULT_FOV_COL,
    x_col_name: str = DEFAULT_X_COL,
    y_col_name: str = DEFAULT_Y_COL,
    ROW_TOL_FACTOR: float = 0.20,
    COL_TOL_FACTOR: float = 0.20
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """
    Map each filename to (row, col) based on stage coordinates.
    Handles rows of different length that are centred or truncated.

    Args:
        filenames: List of filenames for the tiles.
        coords_df: DataFrame with tile coordinates. Must contain columns specified by
                   fov_col_name, x_col_name, y_col_name.
        fov_re: Compiled regular expression to extract FOV identifier from filename.
                The first capturing group should be the FOV identifier.
        fov_col_name: Name of the column in coords_df containing the FOV identifier.
        x_col_name: Name of the column for X coordinates.
        y_col_name: Name of the column for Y coordinates.
        ROW_TOL_FACTOR: Tolerance factor for row clustering (percentage of Y pitch).
        COL_TOL_FACTOR: Tolerance factor for column clustering (percentage of X pitch).

    Returns:
        A tuple: (row_assignments, col_assignments, fname_to_dfidx_map)
        - row_assignments: List of 0-indexed row numbers for each filename.
        - col_assignments: List of 0-indexed column numbers for each filename.
        - fname_to_dfidx_map: Dictionary mapping filename to its original index in coords_df.
    """
    if not filenames:
        logger.info("Received empty filenames list, returning empty results.")
        return [], [], {}

    # --- Validate DataFrame columns ---
    required_cols = [fov_col_name, x_col_name, y_col_name]
    for col in required_cols:
        if col not in coords_df.columns:
            msg = f"Required column '{col}' not found in coords_df."
            logger.error(msg)
            raise KeyError(msg)

    # ----------------------------------------------------------
    # 1.  Collect (x, y) per filename
    # ----------------------------------------------------------
    xy_coords: List[Tuple[float, float]] = []
    fname_to_dfidx_map: Dict[str, int] = {}
    valid_indices_in_filenames = [] # Keep track of original indices of successfully processed filenames

    for i, fname in enumerate(filenames):
        m = fov_re.search(fname)
        if not m:
            logger.warning(f"{fname}: cannot extract FOV with pattern {fov_re.pattern}. Skipping this file.")
            continue
        try:
            fov_str = m.group(1)
            fov = int(fov_str)
        except (IndexError, ValueError):
            logger.warning(f"{fname}: FOV regex matched, but group 1 ('{fov_str}') is not a valid integer. Skipping.")
            continue

        # Ensure fov_col_name in coords_df is of a type comparable to fov
        # This is a simplification; robust type checking might be needed if coords_df is messy
        try:
            df_row = coords_df.loc[coords_df[fov_col_name].astype(int) == fov]
        except ValueError:
            logger.error(f"Could not convert column '{fov_col_name}' to int for comparison. Check data.")
            raise
        except Exception as e:
            logger.error(f"Error accessing fov column '{fov_col_name}': {e}")
            raise


        if df_row.empty:
            logger.warning(f"FOV {fov} (from {fname}) not in coordinates DataFrame. Skipping this file.")
            continue
        if len(df_row) > 1:
            logger.warning(f"FOV {fov} (from {fname}) has multiple entries ({len(df_row)}) in coordinates DataFrame. Using the first one.")

        idx = df_row.index[0]
        fname_to_dfidx_map[fname] = idx # Map original filename to its df index
        try:
            x_val = float(df_row.at[idx, x_col_name])
            y_val = float(df_row.at[idx, y_col_name])
            xy_coords.append((x_val, y_val))
            valid_indices_in_filenames.append(i) # Store original index of this filename
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse coordinates for FOV {fov} (from {fname}) as float: {e}. Skipping.")
            continue
        except KeyError as e: # Should be caught by initial check, but defensive
            logger.error(f"Coordinate column missing for FOV {fov} (from {fname}): {e}. This indicates a problem.")
            # Potentially re-raise or handle more severely if this happens post-initial check
            continue


    if not xy_coords:
        logger.warning("No valid coordinates could be extracted from any filenames. Returning empty results.")
        # Initialize assignments for all original filenames as -1 (unassigned)
        num_original_files = len(filenames)
        return [-1] * num_original_files, [-1] * num_original_files, fname_to_dfidx_map

    x_arr, y_arr = map(np.asarray, zip(*xy_coords)) # These arrays correspond to successfully processed files

    # Initialize full assignment arrays with -1 (unassigned)
    # These will be populated only for files that had valid coordinates
    final_row_assignments = np.full(len(filenames), -1, dtype=int)
    final_col_assignments = np.full(len(filenames), -1, dtype=int)


    # ----------------------------------------------------------
    # 2.  Row clustering
    # ----------------------------------------------------------
    # `sorted_idx_for_xy` are indices into x_arr, y_arr
    sorted_idx_for_xy = np.argsort(y_arr)
    processed_rows: List[RowInfo] = []

    unique_y = np.sort(np.unique(y_arr))
    pitch_y = np.min(np.diff(unique_y)) if len(unique_y) > 1 else 0.0
    if pitch_y == 0.0 and len(unique_y) > 1 : logger.warning("Calculated Y pitch is zero despite multiple unique Y values. Check Y coordinates for precision issues.")

    row_tol = pitch_y * ROW_TOL_FACTOR if pitch_y > MIN_PITCH_FOR_FACTOR else DEFAULT_ABSOLUTE_TOLERANCE
    logger.debug(f"Row clustering: pitch_y={pitch_y:.4f}, row_tol={row_tol:.4f}")

    for gi_xy in sorted_idx_for_xy: # gi_xy is an index for x_arr/y_arr
        y = y_arr[gi_xy]
        if not processed_rows or abs(y - processed_rows[-1].center_y) > row_tol:
            processed_rows.append(RowInfo(center_y=y, tile_indices=[gi_xy]))
        else:
            current_row = processed_rows[-1]
            current_row.tile_indices.append(gi_xy)
            # Update center_y with the new mean
            mean_y = np.mean(y_arr[current_row.tile_indices])
            processed_rows[-1] = RowInfo(center_y=mean_y, tile_indices=current_row.tile_indices)

    # Map row assignments back to original filename indices
    # `temp_row_assignments` are for the subset of successfully processed files
    temp_row_assignments = np.full(len(xy_coords), -1, dtype=int)
    for r_idx, row_info in enumerate(processed_rows):
        if row_info.tile_indices: # Ensure there are members
            # row_info.tile_indices contains indices relative to xy_coords / x_arr / y_arr
            temp_row_assignments[np.array(row_info.tile_indices)] = r_idx


    # ----------------------------------------------------------
    # 3.  Global column clustering
    # ----------------------------------------------------------
    unique_x = np.sort(np.unique(x_arr))
    pitch_x = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 0.0
    if pitch_x == 0.0 and len(unique_x) > 1 : logger.warning("Calculated X pitch is zero despite multiple unique X values. Check X coordinates for precision issues.")


    col_tol = pitch_x * COL_TOL_FACTOR if pitch_x > MIN_PITCH_FOR_FACTOR else DEFAULT_ABSOLUTE_TOLERANCE
    logger.debug(f"Column clustering: pitch_x={pitch_x:.4f}, col_tol={col_tol:.4f}")

    col_centers_list: List[float] = []
    if unique_x.size > 0:
        col_centers_list.append(unique_x[0])
        for x_val in unique_x[1:]:
            if abs(x_val - col_centers_list[-1]) > col_tol:
                col_centers_list.append(x_val)
    col_centers_arr = np.asarray(col_centers_list)

    # `temp_col_assignments` are for the subset of successfully processed files
    temp_col_assignments = np.full(len(xy_coords), -1, dtype=int)
    if x_arr.size > 0:
        if col_centers_arr.size == 0:
            logger.warning("No distinct column centers found. Assigning all processed tiles to column 0.")
            if x_arr.size > 0: temp_col_assignments.fill(0) # Assign all to column 0
        else:
            temp_col_assignments = np.argmin(np.abs(x_arr[:, None] - col_centers_arr[None, :]), axis=1)


    # Populate final_row_assignments and final_col_assignments using valid_indices_in_filenames
    if valid_indices_in_filenames: # Check if any files were processed
        # valid_indices_in_filenames contains the original indices of files that made it into x_arr/y_arr
        # temp_row_assignments and temp_col_assignments are indexed 0..N-1 for N successfully processed files
        final_row_assignments[valid_indices_in_filenames] = temp_row_assignments
        final_col_assignments[valid_indices_in_filenames] = temp_col_assignments

    logger.info(f"Processed {len(xy_coords)}/{len(filenames)} files. Found {len(processed_rows)} rows and {len(col_centers_list)} columns.")
    
    return final_row_assignments.tolist(), final_col_assignments.tolist(), fname_to_dfidx_map



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
        top_displacement = compute_image_overlap2(
            grid[grid["top_ncc_first"] > ncc_threshold], "top", sizeY, sizeX
        )
        overlap_top = np.clip(100 - top_displacement[0] * 100, pou, 100 - pou)
    
    # Always compute left displacement (we know we have left pairs)
    left_displacement = compute_image_overlap2(
        grid[grid["left_ncc_first"] > ncc_threshold], "left", sizeY, sizeX
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
        return pixel_size_um #this is useful metadata for when I integrate OME-
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


def read_coordinates_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read coordinates from CSV file.
    
    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to coordinates CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing coordinates
    """
    return pd.read_csv(csv_path)


def read_tiff_images(directory: Union[str, Path], pattern: str) -> Dict[str, np.ndarray]:
    """Read TIFF images from directory.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images
    pattern : str
        Glob pattern to match TIFF files
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping filenames to image arrays
    """
    directory = Path(directory)
    images = {}
    
    for tiff_path in directory.glob(pattern):
        try:
            images[tiff_path.name] = tifffile.imread(str(tiff_path))
        except Exception as e:
            print(f"Error reading {tiff_path}: {e}")
            continue
            
    return images


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