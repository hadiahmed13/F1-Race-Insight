"""Streamlit dashboard for F1 Race Insight."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import fastf1
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from plotly.subplots import make_subplots

from src.config import (DASHBOARD_CACHE_TTL, LATEST_MODEL_PATH, PIPELINE_PATH,
                      PROCESSED_DATASET_PATH)
from src.models.predict import predict_batch
from src.utils.io import download_github_asset, load_dataframe
from src.utils.logging import get_logger

# Constants
DASHBOARD_CACHE_TTL = 3600  # 1 hour cache TTL
MODEL_PATH = os.path.join("models", "checkpoints", "samples", "model_sample.joblib")
PIPELINE_PATH = os.path.join("models", "checkpoints", "samples", "pipeline_sample.joblib")
SAMPLE_DATA_PATH = os.path.join("data", "processed", "samples", "f1_safety_car_dataset_sample.parquet")

# Set FastF1 options
fastf1.Cache.enable_cache("./data/raw/fastf1_cache")

# Initialize logger
logger = get_logger(__name__)


@st.cache_resource(ttl=DASHBOARD_CACHE_TTL)
def load_model_and_pipeline():
    """Load the model and pipeline from files.

    Returns:
        Tuple of (model, pipeline).
    """
    # Try to load locally
    try:
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        return model, pipeline
    except (FileNotFoundError, OSError) as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None


@st.cache_resource(ttl=DASHBOARD_CACHE_TTL)
def load_dataset():
    """Load the dataset from file.

    Returns:
        The loaded DataFrame or None if not found.
    """
    # Try to load sample data
    try:
        return pd.read_parquet(SAMPLE_DATA_PATH)
    except FileNotFoundError as e:
        st.error(f"Failed to load dataset: {str(e)}")
        return None


@st.cache_data(ttl=DASHBOARD_CACHE_TTL)
def get_race_options(df: pd.DataFrame) -> list:
    """Get the list of available races.

    Args:
        df: The dataset DataFrame.

    Returns:
        List of tuples with (race_id, display_name).
    """
    if df is None:
        return []
    
    # Extract unique races
    races = df["RaceId"].unique()
    
    # Create display names
    race_options = []
    for race_id in races:
        display_name = f"Race: {race_id}"
        race_options.append((race_id, display_name))
    
    # Sort by race_id
    race_options.sort(key=lambda x: x[0], reverse=True)
    
    return race_options


@st.cache_data(ttl=DASHBOARD_CACHE_TTL)
def get_race_data(df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Get data for a specific race.

    Args:
        df: The full dataset.
        race_id: The race ID to filter by.

    Returns:
        DataFrame with data for the specified race.
    """
    if df is None:
        return pd.DataFrame()
    
    race_df = df[df["RaceId"] == race_id].copy()
    
    # Sort by lap number
    race_df = race_df.sort_values("LapNumber")
    
    return race_df


@st.cache_data(ttl=DASHBOARD_CACHE_TTL)
def predict_race_risks(race_df: pd.DataFrame) -> pd.DataFrame:
    """Predict safety car risks for a race.

    Args:
        race_df: DataFrame with race data.

    Returns:
        DataFrame with added risk predictions.
    """
    if race_df.empty:
        return pd.DataFrame()
    
    try:
        # Make predictions
        race_with_preds = predict_batch(race_df)
        
        # Calculate average risk per lap
        lap_risks = (
            race_with_preds.groupby("LapNumber")["sc_probability"]
            .mean()
            .reset_index()
        )
        
        # Add safety car status info
        lap_safety_car = (
            race_with_preds.groupby("LapNumber")["SafetyCar"]
            .any()
            .reset_index()
        )
        
        lap_risks = lap_risks.merge(lap_safety_car, on="LapNumber")
        
        return lap_risks
    
    except Exception as e:
        st.error(f"Error predicting race risks: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_CACHE_TTL)
def get_race_session(race_id: str):
    """Load the FastF1 session for a race.

    Args:
        race_id: The race ID.

    Returns:
        The FastF1 session or None if not found.
    """
    try:
        # Extract year, round, and session type from race_id
        parts = race_id.split("_")
        year = int(parts[0])
        round_num = int(parts[1])
        session_type = "R"  # Assuming race session
        
        # Load the session
        session = fastf1.get_session(year, round_num, session_type)
        session.load()
        return session
    
    except Exception as e:
        st.warning(f"Could not load FastF1 session: {str(e)}")
        return None


def plot_lap_risks(lap_risks: pd.DataFrame) -> go.Figure:
    """Create a plotly figure of lap-by-lap safety car risks.

    Args:
        lap_risks: DataFrame with lap risks.

    Returns:
        Plotly figure.
    """
    if lap_risks.empty:
        return go.Figure()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add risk line
    fig.add_trace(
        go.Scatter(
            x=lap_risks["LapNumber"],
            y=lap_risks["sc_probability"],
            mode="lines+markers",
            name="Safety Car Risk",
            line=dict(color="red", width=3),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    
    # Add safety car indicator
    if "SafetyCar" in lap_risks.columns:
        sc_laps = lap_risks[lap_risks["SafetyCar"]]
        if not sc_laps.empty:
            fig.add_trace(
                go.Scatter(
                    x=sc_laps["LapNumber"],
                    y=[1] * len(sc_laps),
                    mode="markers",
                    name="Safety Car Deployed",
                    marker=dict(
                        symbol="square",
                        size=15,
                        color="yellow",
                        line=dict(width=2, color="black"),
                    ),
                ),
                secondary_y=True,
            )
    
    # Update layout
    fig.update_layout(
        title="Safety Car Deployment Risk by Lap",
        xaxis_title="Lap Number",
        yaxis_title="Risk Probability",
        legend=dict(y=0.99, x=0.01, orientation="h"),
        hovermode="x unified",
        height=400,
    )
    
    # Set y-axis range
    fig.update_yaxes(range=[0, 1], secondary_y=False)
    fig.update_yaxes(
        range=[0, 1.1],
        secondary_y=True,
        showticklabels=False,
        showgrid=False,
    )
    
    # Add threshold line
    threshold = lap_risks["threshold"].iloc[0] if "threshold" in lap_risks.columns else 0.5
    fig.add_shape(
        type="line",
        x0=lap_risks["LapNumber"].min(),
        x1=lap_risks["LapNumber"].max(),
        y0=threshold,
        y1=threshold,
        line=dict(color="gray", width=2, dash="dash"),
        xref="x",
        yref="y",
    )
    
    # Add threshold annotation
    fig.add_annotation(
        x=lap_risks["LapNumber"].min(),
        y=threshold,
        text=f"Threshold: {threshold:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255, 255, 255, 0.7)",
    )
    
    return fig


def plot_track_map(session, lap_risks: pd.DataFrame, current_lap: int) -> Optional[plt.Figure]:
    """Plot the track map with color-coded risk indicators.

    Args:
        session: FastF1 session.
        lap_risks: DataFrame with lap risks.
        current_lap: The currently selected lap.

    Returns:
        Matplotlib figure or None if plotting fails.
    """
    if session is None or lap_risks.empty:
        return None
    
    try:
        # Create a new matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the circuit
        session.plot_track(ax=ax)
        
        # Get the risk for the current lap
        current_risk = None
        if current_lap in lap_risks["LapNumber"].values:
            current_risk = lap_risks.loc[
                lap_risks["LapNumber"] == current_lap, "sc_probability"
            ].iloc[0]
        
        # Update the title to include the risk
        risk_text = f"- Safety Car Risk: {current_risk:.2f}" if current_risk is not None else ""
        ax.set_title(f"Track Map for Lap {current_lap} {risk_text}")
        
        # Add colorbar for reference
        if current_risk is not None:
            # Create a color scale from green to red
            cmap = plt.cm.RdYlGn_r
            norm = plt.Normalize(0, 1)
            
            # Create a ScalarMappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Add the colorbar
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Safety Car Deployment Risk")
            
            # Add a marker at the current risk level
            if current_risk is not None:
                cbar.ax.plot([0, 1], [current_risk, current_risk], "k-", lw=2)
        
        return fig
    
    except Exception as e:
        st.warning(f"Error plotting track map: {str(e)}")
        return None


def get_shap_explanation(
    model, pipeline, race_df: pd.DataFrame, current_lap: int
) -> Optional[plt.Figure]:
    """Generate SHAP explanation for the current lap.

    Args:
        model: Trained model.
        pipeline: Feature processing pipeline.
        race_df: DataFrame with race data.
        current_lap: The currently selected lap.

    Returns:
        Matplotlib figure with SHAP explanation or None if fails.
    """
    if model is None or pipeline is None or race_df.empty:
        return None
    
    try:
        # Filter data for the current lap
        current_lap_data = race_df[race_df["LapNumber"] == current_lap]
        
        if current_lap_data.empty:
            return None
        
        # Select the first driver's data (simplification)
        instance = current_lap_data.iloc[0:1].copy()
        
        # Transform the features
        feature_names = list(pipeline.feature_names_in_)
        available_features = [col for col in feature_names if col in instance.columns]
        X = pipeline.transform(instance[available_features])
        
        # Create the SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Create a figure for the SHAP values
        fig = plt.figure(figsize=(12, 8))
        
        # Plot the SHAP values
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.title(f"SHAP Feature Importance for Lap {current_lap}")
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.warning(f"Error generating SHAP explanation: {str(e)}")
        return None


def main():
    """Main function for the Streamlit dashboard."""
    # Page config
    st.set_page_config(
        page_title="F1 Race Insight - Safety Car Predictor",
        page_icon="🏎️",
        layout="wide",
    )
    
    # Header
    st.title("🏎️ F1 Race Insight - Safety Car Predictor")
    st.markdown(
        """
        This dashboard shows predictions for safety car deployment in Formula 1 races.
        Select a race and lap to see the predicted risk of a safety car being deployed.
        """
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_dataset()
        model, pipeline = load_model_and_pipeline()
    
    if df is None:
        st.error(
            "Failed to load the dataset. Please check that the sample data exists."
        )
        return
    
    # Sidebar - Race selector
    st.sidebar.header("Race Selection")
    
    race_options = get_race_options(df)
    if not race_options:
        st.error("No races found in the dataset.")
        return
    
    race_display_dict = {race_id: display for race_id, display in race_options}
    
    selected_race_display = st.sidebar.selectbox(
        "Select Race", options=list(race_display_dict.values())
    )
    
    # Find the race_id from the display name
    selected_race_id = next(
        (race_id for race_id, display in race_options if display == selected_race_display),
        None,
    )
    
    if not selected_race_id:
        st.error("Invalid race selection")
        return
    
    # Get data for the selected race
    race_df = get_race_data(df, selected_race_id)
    
    if race_df.empty:
        st.error(f"No data found for race: {selected_race_display}")
        return
    
    # Get the lap risks
    lap_risks = predict_race_risks(race_df)
    
    # Sidebar - Lap selector
    min_lap = int(race_df["LapNumber"].min())
    max_lap = int(race_df["LapNumber"].max())
    
    current_lap = st.sidebar.slider(
        "Select Lap", min_value=min_lap, max_value=max_lap, value=min_lap
    )
    
    # Load FastF1 session
    session = get_race_session(selected_race_id)
    
    # Main area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Lap-by-lap risk chart
        st.subheader("Lap-by-Lap Safety Car Risk")
        risk_fig = plot_lap_risks(lap_risks)
        st.plotly_chart(risk_fig, use_container_width=True)
        
        # Display race info
        race_info = {
            "Track": race_df["TrackId"].iloc[0],
            "Year": race_df["Year"].iloc[0],
            "Track Length": f"{race_df['TrackLength'].iloc[0]:.3f} km",
            "Corners": int(race_df["Corners"].iloc[0]),
            "Weather": f"{race_df['AirTemp'].median():.1f}°C, Humidity: {race_df['Humidity'].median():.1f}%",
            "Rainfall": f"{race_df['Rainfall'].median():.2f} mm",
        }
        
        st.subheader("Race Information")
        info_cols = st.columns(3)
        
        for i, (key, value) in enumerate(race_info.items()):
            col_idx = i % 3
            info_cols[col_idx].metric(key, value)
    
    with col2:
        # Track map
        st.subheader("Track Map")
        track_fig = plot_track_map(session, lap_risks, current_lap)
        if track_fig:
            st.pyplot(track_fig)
        else:
            st.info("Track map not available for this race")
    
    # SHAP explanation - below both columns
    st.subheader(f"Feature Importance for Lap {current_lap}")
    shap_fig = get_shap_explanation(model, pipeline, race_df, current_lap)
    if shap_fig:
        st.pyplot(shap_fig)
    else:
        st.info("SHAP explanation not available for this lap")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **F1 Race Insight Safety Car Predictor** - Predicting safety car deployments in Formula 1 races using machine learning.
        
        Data source: [FastF1](https://github.com/theOehrly/Fast-F1)
        """
    )


if __name__ == "__main__":
    main() 