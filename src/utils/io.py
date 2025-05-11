"""I/O utilities for F1 Race Insight."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests
from requests.exceptions import RequestException

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        path: The directory path to check/create.

    Returns:
        The Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_dataframe(
    df: pd.DataFrame, filepath: Union[str, Path], index: bool = False
) -> None:
    """Save a DataFrame to a Parquet file.

    Args:
        df: The DataFrame to save.
        filepath: The path to save the file to.
        index: Whether to include the DataFrame index in the saved file.
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    
    logger.info(f"Saving DataFrame to {filepath}", shape=df.shape)
    df.to_parquet(filepath, index=index)


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file.

    Args:
        filepath: The path of the file to load.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading DataFrame from {filepath}")
    return pd.read_parquet(filepath)


def download_github_asset(
    repo: str, tag: str, asset_name: str, output_path: Optional[Path] = None
) -> Path:
    """Download an asset from a GitHub release.

    Args:
        repo: The GitHub repository (e.g., "hadiahmed13/f1-race-insight").
        tag: The release tag (e.g., "v1.0.0").
        asset_name: The name of the asset to download.
        output_path: The path to save the downloaded file to.
            If None, saves to the raw data directory.

    Returns:
        The path where the asset was saved.

    Raises:
        RequestException: If the download fails.
    """
    if output_path is None:
        output_path = RAW_DATA_DIR / asset_name
    
    # Ensure the output directory exists
    ensure_dir_exists(output_path.parent)
    
    # Construct the GitHub API URL for the release
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    
    try:
        # Get the release information
        logger.info(f"Fetching release info for {repo}:{tag}")
        release_response = requests.get(api_url)
        release_response.raise_for_status()
        release_data = release_response.json()
        
        # Find the specified asset
        asset_url = None
        for asset in release_data["assets"]:
            if asset["name"] == asset_name:
                asset_url = asset["browser_download_url"]
                break
        
        if not asset_url:
            raise ValueError(f"Asset '{asset_name}' not found in release {tag}")
        
        # Download the asset
        logger.info(f"Downloading asset {asset_name} to {output_path}")
        with requests.get(asset_url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return output_path
    
    except RequestException as e:
        logger.error(f"Failed to download asset {asset_name}", error=str(e))
        raise 