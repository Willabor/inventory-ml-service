"""Feature engineering for transfer prediction model."""
import pandas as pd
import numpy as np
from typing import Tuple


def prepare_transfer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert raw transfer data into engineered features for TabPFN.

    Args:
        df: Raw data from data extraction

    Returns:
        Tuple of (features DataFrame, target Series)
    """

    # Create feature DataFrame
    features = pd.DataFrame()

    # Velocity features
    features['velocity_ratio'] = df['to_store_daily_sales'] / (df['from_store_daily_sales'] + 0.01)
    features['velocity_gap'] = df['to_store_daily_sales'] - df['from_store_daily_sales']
    features['from_velocity'] = df['from_store_daily_sales']
    features['to_velocity'] = df['to_store_daily_sales']

    # Stock features
    features['stock_ratio'] = df['to_store_qty'] / (df['from_store_qty'] + 1)
    features['from_stock'] = np.log1p(df['from_store_qty'])  # Log transform for normalization
    features['to_stock'] = np.log1p(df['to_store_qty'])

    # Days of supply features
    features['from_days_supply'] = df['from_store_qty'] / (df['from_store_daily_sales'] + 0.01)
    features['to_days_supply'] = df['to_store_qty'] / (df['to_store_daily_sales'] + 0.01)
    features['days_supply_gap'] = features['from_days_supply'] - features['to_days_supply']

    # Pricing features
    features['margin_percent'] = df['margin_percent']
    features['avg_price'] = df['avg_selling_price']
    features['price_tier'] = pd.cut(
        df['avg_selling_price'],
        bins=[0, 20, 50, 100, 500, float('inf')],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Category encoding (one-hot)
    if 'category' in df.columns:
        category_dummies = pd.get_dummies(df['category'], prefix='cat', dtype=float)
        # Limit to top 10 categories to avoid too many features
        if len(category_dummies.columns) > 10:
            top_cats = category_dummies.sum().nlargest(10).index
            category_dummies = category_dummies[top_cats]
        features = pd.concat([features, category_dummies], axis=1)

    # Store encoding (one-hot)
    from_store_dummies = pd.get_dummies(df['from_store'], prefix='from', dtype=float)
    to_store_dummies = pd.get_dummies(df['to_store'], prefix='to', dtype=float)
    features = pd.concat([features, from_store_dummies, to_store_dummies], axis=1)

    # Interaction features
    features['velocity_x_margin'] = features['velocity_ratio'] * features['margin_percent']
    features['stock_x_velocity'] = features['stock_ratio'] * features['velocity_ratio']

    # Handle missing values
    features = features.fillna(0)

    # Cap extreme values (outliers)
    features['velocity_ratio'] = features['velocity_ratio'].clip(0, 20)
    features['from_days_supply'] = features['from_days_supply'].clip(0, 365)
    features['to_days_supply'] = features['to_days_supply'].clip(0, 365)

    # Extract target if available
    target = df['transfer_success'] if 'transfer_success' in df.columns else None

    return features, target


def calculate_recommended_quantity(
    df: pd.DataFrame,
    success_probability: np.ndarray,
    target_days: int = 14
) -> np.ndarray:
    """
    Calculate recommended transfer quantity based on ML predictions.

    Args:
        df: Original data with stock and velocity info
        success_probability: Model's prediction probabilities
        target_days: Target days of supply for TO store

    Returns:
        Array of recommended transfer quantities
    """

    recommended = []

    for idx, row in df.iterrows():
        prob = success_probability[idx]

        # Only recommend if probability of success is high enough
        if prob < 0.5:
            recommended.append(0)
            continue

        # Calculate based on TO store velocity and target days
        to_velocity = row['to_store_daily_sales']
        from_qty = row['from_store_qty']

        # Target quantity: enough to cover target_days at TO store
        target_qty = int(np.ceil(to_velocity * target_days))

        # Don't transfer more than 50% of FROM store stock
        max_qty = int(from_qty / 2)

        # Scale by confidence (lower probability = lower quantity)
        confidence_factor = (prob - 0.5) / 0.5  # 0.5 -> 0, 1.0 -> 1.0
        recommended_qty = int(min(target_qty, max_qty) * (0.5 + 0.5 * confidence_factor))

        # Cap at 20 units max
        recommended_qty = min(recommended_qty, 20)

        # Minimum of 1 if we're recommending at all
        recommended_qty = max(recommended_qty, 1) if recommended_qty > 0 else 0

        recommended.append(recommended_qty)

    return np.array(recommended)


def calculate_ml_priority(
    success_probability: np.ndarray,
    margin_percent: pd.Series,
    velocity_gap: pd.Series
) -> np.ndarray:
    """
    Calculate ML-based priority score for transfers.

    Args:
        success_probability: Model's success predictions
        margin_percent: Profit margin percentages
        velocity_gap: Sales velocity difference (to - from)

    Returns:
        Array of priority scores (higher = better)
    """

    # Normalize inputs to 0-1 range
    prob_norm = success_probability
    margin_norm = np.clip(margin_percent / 100, 0, 1)
    velocity_norm = np.clip(velocity_gap / velocity_gap.max(), 0, 1) if velocity_gap.max() > 0 else 0

    # Weighted combination
    priority_score = (
        prob_norm * 0.5 +        # 50% weight on success probability
        margin_norm * 0.3 +       # 30% weight on margin
        velocity_norm * 0.2       # 20% weight on velocity gap
    )

    return priority_score


def assign_priority_label(priority_score: np.ndarray) -> np.ndarray:
    """
    Convert continuous priority scores to High/Medium/Low labels.

    Args:
        priority_score: Continuous priority scores

    Returns:
        Array of priority labels
    """

    labels = []
    for score in priority_score:
        if score >= 0.7:
            labels.append('High')
        elif score >= 0.5:
            labels.append('Medium')
        else:
            labels.append('Low')

    return np.array(labels)
