"""Target feature creation for F1 Race Insight."""

import pandas as pd

from src.config import TARGET_COLUMN
from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_safety_car_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add safety car deployment target feature.

    This function creates a binary target variable indicating whether
    a safety car was deployed on the next lap.

    Args:
        df: DataFrame with lap data.

    Returns:
        DataFrame with added target column.
    """
    logger.info("Adding safety car target feature")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Initialize target column
    result_df[TARGET_COLUMN] = 0
    
    # Process each race
    for race_id in result_df["RaceId"].unique():
        race_data = result_df[result_df["RaceId"] == race_id]
        
        # Process each lap
        for lap_number in range(1, race_data["LapNumber"].max()):
            current_lap = race_data[race_data["LapNumber"] == lap_number]
            next_lap = race_data[race_data["LapNumber"] == lap_number + 1]
            
            if next_lap.empty:
                continue
            
            # Check if there's a safety car deployment in the next lap
            if (
                (next_lap["SafetyCar"].any()) and 
                # Make sure it's a new deployment (not already active in current lap)
                (not current_lap["SafetyCar"].any())
            ):
                # Set target to 1 for all drivers in the current lap
                result_df.loc[
                    (result_df["RaceId"] == race_id) & 
                    (result_df["LapNumber"] == lap_number),
                    TARGET_COLUMN
                ] = 1
    
    # Log statistics
    positive_count = result_df[TARGET_COLUMN].sum()
    total_count = len(result_df)
    positive_pct = 100 * positive_count / total_count if total_count > 0 else 0
    
    logger.info(
        f"Safety car target added",
        positive_examples=positive_count,
        total_examples=total_count,
        positive_percentage=f"{positive_pct:.2f}%",
    )
    
    return result_df 