"""FastF1 data downloader for F1 Race Insight."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fastf1
import pandas as pd
from fastf1.core import Session

from src.config import DATA_YEARS, FASTF1_CACHE_DIR, RAW_DATA_DIR
from src.utils.io import ensure_dir_exists, save_dataframe
from src.utils.logging import get_logger, log_function_call

# Initialize logger
logger = get_logger(__name__)

# Initialize FastF1 cache
ensure_dir_exists(FASTF1_CACHE_DIR)
fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))


def get_session_types() -> List[str]:
    """Get the list of session types to download.

    Returns:
        List of session types.
    """
    return ["Q", "R"]  # Qualifying and Race only


def get_completed_rounds(year: int) -> List[int]:
    """Get the list of completed rounds for a given year.

    Args:
        year: The F1 season year.

    Returns:
        List of completed round numbers.
    """
    try:
        logger.info(f"Getting completed rounds for {year}")
        schedule = fastf1.get_event_schedule(year)
        
        # Filter out future races
        completed_events = schedule[schedule["EventDate"] < pd.Timestamp.now()]
        
        # Get the round numbers
        rounds = completed_events["RoundNumber"].tolist()
        
        logger.info(f"Found {len(rounds)} completed rounds for {year}: {rounds}")
        return rounds
    
    except Exception as e:
        logger.error(f"Error getting completed rounds for {year}", error=str(e))
        return []


def download_session(
    year: int, round_num: int, session_type: str
) -> Optional[Tuple[Session, str]]:
    """Download session data using FastF1.

    Args:
        year: The F1 season year.
        round_num: The round number.
        session_type: The session type (e.g., "Q" for qualifying, "R" for race).

    Returns:
        Tuple of (session object, session name) if successful, None otherwise.
    """
    try:
        log_function_call(
            logger,
            "download_session",
            {"year": year, "round": round_num, "session_type": session_type},
        )
        
        # Load the session
        session = fastf1.get_session(year, round_num, session_type)
        session.load()
        
        # Get the event name
        event_name = session.event["EventName"].replace(" ", "_")
        
        logger.info(
            f"Successfully loaded session",
            year=year,
            round=round_num,
            session=session_type,
            event=event_name,
        )
        
        return session, event_name
    
    except Exception as e:
        logger.error(
            f"Error downloading session",
            year=year,
            round=round_num,
            session=session_type,
            error=str(e),
        )
        return None


def process_session(
    session: Session, year: int, round_num: int, session_type: str, event_name: str
) -> None:
    """Process and save session data.

    Args:
        session: The FastF1 session object.
        year: The F1 season year.
        round_num: The round number.
        session_type: The session type.
        event_name: The event name.
    """
    try:
        # Create output filename
        output_filename = f"{year}_{round_num:02d}_{event_name}_{session_type}"
        
        # Get and save lap data
        laps_df = session.laps
        
        # Add session info
        laps_df["Year"] = year
        laps_df["Round"] = round_num
        laps_df["EventName"] = event_name
        laps_df["SessionType"] = session_type
        
        # Save lap data
        save_dataframe(laps_df, RAW_DATA_DIR / f"{output_filename}_laps.parquet")
        
        # Get and save weather data if available
        try:
            weather_df = session.weather_data
            weather_df["Year"] = year
            weather_df["Round"] = round_num
            weather_df["EventName"] = event_name
            weather_df["SessionType"] = session_type
            save_dataframe(weather_df, RAW_DATA_DIR / f"{output_filename}_weather.parquet")
        except Exception as e:
            logger.warning(f"Weather data not available", session=output_filename, error=str(e))
        
        # Save car setup data if available
        try:
            car_data_by_driver = {}
            for driver in session.drivers:
                try:
                    car_data = session.car_data[driver]
                    car_data["Driver"] = driver
                    car_data["Year"] = year
                    car_data["Round"] = round_num
                    car_data["EventName"] = event_name
                    car_data["SessionType"] = session_type
                    car_data_by_driver[driver] = car_data
                except Exception as e_driver:
                    logger.warning(
                        f"Car data not available for driver",
                        driver=driver,
                        session=output_filename,
                        error=str(e_driver),
                    )
            
            if car_data_by_driver:
                car_data_combined = pd.concat(car_data_by_driver.values())
                save_dataframe(
                    car_data_combined, RAW_DATA_DIR / f"{output_filename}_car_data.parquet"
                )
        except Exception as e:
            logger.warning(f"Car data not available", session=output_filename, error=str(e))
        
        # Save track status data if available
        try:
            track_status_df = session.track_status
            track_status_df["Year"] = year
            track_status_df["Round"] = round_num
            track_status_df["EventName"] = event_name
            track_status_df["SessionType"] = session_type
            save_dataframe(
                track_status_df, RAW_DATA_DIR / f"{output_filename}_track_status.parquet"
            )
        except Exception as e:
            logger.warning(f"Track status data not available", session=output_filename, error=str(e))
        
        logger.info(f"Successfully processed and saved session data", session=output_filename)
    
    except Exception as e:
        logger.error(
            f"Error processing session",
            year=year,
            round=round_num,
            session=session_type,
            error=str(e),
        )


def download_all_sessions() -> None:
    """Download all sessions for the configured years."""
    # Get already downloaded sessions to avoid re-downloading
    existing_files = set(f.stem for f in RAW_DATA_DIR.glob("*_laps.parquet"))
    logger.info(f"Found {len(existing_files)} existing files")
    
    # Download sessions for each year
    for year in DATA_YEARS:
        rounds = get_completed_rounds(year)
        for round_num in rounds:
            for session_type in get_session_types():
                # Check if this session has already been downloaded
                session_key = f"{year}_{round_num:02d}_*_{session_type}_laps"
                if any(session_key.replace("*", "") in f for f in existing_files):
                    logger.info(
                        f"Skipping already downloaded session",
                        year=year,
                        round=round_num,
                        session=session_type,
                    )
                    continue
                
                # Download and process the session
                result = download_session(year, round_num, session_type)
                if result:
                    session, event_name = result
                    process_session(session, year, round_num, session_type, event_name)


if __name__ == "__main__":
    logger.info("Starting FastF1 data download")
    download_all_sessions()
    logger.info("Completed FastF1 data download") 