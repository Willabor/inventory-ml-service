"""
Product Segmentation Predictor using ML clustering and classification.

This model combines:
1. K-Means clustering for customer/product behavior patterns
2. Random Forest for segment classification
3. Feature engineering for RFM (Recency, Frequency, Monetary) analysis
"""
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score

from utils.database import db
from config import settings


class SegmentationPredictor:
    """ML-powered product segmentation for Google Marketing campaigns."""

    def __init__(
        self,
        new_arrivals_days: int = 60,
        best_seller_threshold: int = 50,
        core_high_threshold: int = 40,
        core_medium_threshold: int = 20,
        core_low_threshold: int = 6,
        clearance_days: int = 180,
        filters: Optional[Dict] = None
    ):
        self.cluster_model: Optional[KMeans] = None
        self.classification_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_version: str = ""
        self.training_date: Optional[datetime] = None
        self.metrics: Dict = {}

        # Configurable classification thresholds
        self.new_arrivals_days = new_arrivals_days
        self.best_seller_threshold = best_seller_threshold
        self.core_high_threshold = core_high_threshold
        self.core_medium_threshold = core_medium_threshold
        self.core_low_threshold = core_low_threshold
        self.clearance_days = clearance_days

        # Data filters
        self.filters = filters or {}

    def _build_where_clauses(self) -> List[str]:
        """Build SQL WHERE clauses from filters."""
        where_clauses = []

        # Category filters
        if self.filters.get('includedCategories') and len(self.filters['includedCategories']) > 0:
            categories = ', '.join([f"'{cat}'" for cat in self.filters['includedCategories']])
            where_clauses.append(f"i.category IN ({categories})")

        # Vendor filters (exclusion)
        if self.filters.get('excludedVendors') and len(self.filters['excludedVendors']) > 0:
            vendors = ', '.join([f"'{v}'" for v in self.filters['excludedVendors']])
            where_clauses.append(f"i.vendor_name NOT IN ({vendors})")

        # Gender filters
        if self.filters.get('includedGenders') and len(self.filters['includedGenders']) > 0:
            genders = ', '.join([f"'{g}'" for g in self.filters['includedGenders']])
            where_clauses.append(f"i.gender IN ({genders})")

        # Price filters
        if self.filters.get('minPrice', 0) > 0:
            where_clauses.append(f"i.selling_price >= {self.filters['minPrice']}")
        if self.filters.get('maxPrice', 99999) < 99999:
            where_clauses.append(f"i.selling_price <= {self.filters['maxPrice']}")

        # Inventory filters
        if self.filters.get('excludeZeroInventory', True):
            where_clauses.append("(COALESCE(i.gm_qty, 0) + COALESCE(i.hm_qty, 0) + COALESCE(i.nm_qty, 0) + COALESCE(i.lm_qty, 0) + COALESCE(i.hq_qty, 0)) > 0")

        if self.filters.get('minInventory', 0) > 0:
            where_clauses.append(f"(COALESCE(i.gm_qty, 0) + COALESCE(i.hm_qty, 0) + COALESCE(i.nm_qty, 0) + COALESCE(i.lm_qty, 0) + COALESCE(i.hq_qty, 0)) >= {self.filters['minInventory']}")

        if self.filters.get('maxInventory', 99999) < 99999:
            where_clauses.append(f"(COALESCE(i.gm_qty, 0) + COALESCE(i.hm_qty, 0) + COALESCE(i.nm_qty, 0) + COALESCE(i.lm_qty, 0) + COALESCE(i.hq_qty, 0)) <= {self.filters['maxInventory']}")

        return where_clauses

    def _build_store_sum_clause(self) -> str:
        """Build SUM clause for store quantities based on included stores."""
        included_stores = self.filters.get('includedStores', ['GM', 'HM', 'NM', 'LM', 'HQ'])
        store_mapping = {
            'GM': 'i.gm_qty',
            'HM': 'i.hm_qty',
            'NM': 'i.nm_qty',
            'LM': 'i.lm_qty',
            'HQ': 'i.hq_qty'
        }

        included_columns = [f"COALESCE({store_mapping[store]}, 0)" for store in included_stores if store in store_mapping]

        if not included_columns:
            # Default to all stores if none selected
            included_columns = ["COALESCE(i.gm_qty, 0)", "COALESCE(i.hm_qty, 0)", "COALESCE(i.nm_qty, 0)", "COALESCE(i.lm_qty, 0)", "COALESCE(i.hq_qty, 0)"]

        return " + ".join(included_columns)

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract RFM + inventory features for segmentation.

        Features:
        - Recency: Days since last sale/receive
        - Frequency: Number of sales/receives
        - Monetary: Revenue, margin, inventory value
        - Inventory: Stock levels, days of supply
        - Sales: Velocity, 30d/90d sales
        """
        features = pd.DataFrame()

        # Recency features
        features['days_since_last_sale'] = data.get('days_since_last_sale', 0).fillna(999)
        features['days_since_last_receive'] = data.get('days_since_last_receive', 0).fillna(999)

        # Frequency features
        features['receive_count'] = data.get('receive_count', 0).fillna(0)
        features['units_sold_30d'] = data.get('units_sold_30d', 0).fillna(0)
        features['units_sold_90d'] = data.get('units_sold_90d', 0).fillna(0)

        # Monetary features
        features['avg_selling_price'] = data.get('avg_selling_price', 0).fillna(0)
        features['avg_margin_percent'] = data.get('avg_margin_percent', 0).fillna(0)
        features['inventory_value'] = data.get('inventory_value', 0).fillna(0)
        features['margin_per_unit'] = data.get('margin_per_unit', 0).fillna(0)

        # Inventory features
        features['total_active_qty'] = data.get('total_active_qty', 0).fillna(0)
        features['sales_velocity'] = data.get('sales_velocity', 0).fillna(0)

        # Computed ratios
        features['revenue_per_unit'] = features['avg_selling_price']
        features['margin_to_price_ratio'] = (
            features['avg_margin_percent'] / 100.0
        )
        features['turnover_rate'] = np.where(
            features['total_active_qty'] > 0,
            features['units_sold_30d'] / features['total_active_qty'],
            0
        )

        # Stock status encoding
        features['is_dead_stock'] = (features['days_since_last_sale'] > 180).astype(int)
        features['is_new_arrival'] = (features['days_since_last_receive'] <= 60).astype(int)
        features['is_high_frequency'] = (features['receive_count'] >= 40).astype(int)

        # Receiving history features (optional)
        if 'receiving_frequency' in data.columns:
            features['receiving_frequency'] = data.get('receiving_frequency', 0).fillna(0)
            features['total_receives'] = data.get('total_receives', 0).fillna(0)
            features['avg_receiving_cost'] = data.get('avg_receiving_cost', 0).fillna(0)
            features['cost_volatility'] = data.get('cost_volatility', 0).fillna(0)
            features['reversal_rate'] = data.get('reversal_rate', 0).fillna(0)
            features['avg_days_between_receives'] = data.get('avg_days_between_receives', 999).fillna(999)
            features['days_since_last_receiving'] = data.get('days_since_last_receiving', 999).fillna(999)

            # Receiving-based computed features
            features['is_frequent_restock'] = (features['receiving_frequency'] >= 5).astype(int)
            features['has_cost_volatility'] = (features['cost_volatility'] > 2.0).astype(int)
            features['high_reversal_risk'] = (features['reversal_rate'] > 0.1).astype(int)

        # Filter features if specified
        selected_features = self.filters.get('selectedFeatures', [])
        if selected_features and len(selected_features) > 0:
            # Only keep features that are in the selected list
            available_features = [col for col in features.columns if col in selected_features]
            if len(available_features) > 0:
                features = features[available_features]
                print(f"Filtered to {len(available_features)} selected features: {', '.join(available_features[:5])}{'...' if len(available_features) > 5 else ''}")
            else:
                print("Warning: No selected features found in feature set, using all features")

        return features

    def assign_rule_based_segments(self, data: pd.DataFrame) -> pd.Series:
        """
        Assign initial segment labels based on business rules (with configurable thresholds).
        These will be used for supervised training.
        """
        segments = pd.Series(['Unknown'] * len(data), index=data.index)

        # Best Sellers: High sales + high frequency (configurable threshold)
        best_seller_mask = (
            (data['receive_count'] >= self.best_seller_threshold) &
            (data['units_sold_30d'] >= data['units_sold_30d'].quantile(0.9)) &
            (data['sales_velocity'] > 0.5)
        )
        segments[best_seller_mask] = 'Best Seller'

        # Core High: Very high receive count (configurable threshold)
        core_high_mask = (
            (data['receive_count'] >= self.core_high_threshold) &
            (~best_seller_mask)
        )
        segments[core_high_mask] = 'Core High'

        # Core Medium: Medium-high receive count (configurable threshold)
        core_medium_mask = (
            (data['receive_count'] >= self.core_medium_threshold) &
            (data['receive_count'] < self.core_high_threshold) &
            (~best_seller_mask)
        )
        segments[core_medium_mask] = 'Core Medium'

        # Core Low: Low receive count but established (configurable threshold)
        core_low_mask = (
            (data['receive_count'] >= self.core_low_threshold) &
            (data['receive_count'] < self.core_medium_threshold) &
            (~best_seller_mask)
        )
        segments[core_low_mask] = 'Core Low'

        # New Arrivals: Recently received (configurable days window)
        new_arrival_mask = (
            (data['days_since_last_receive'] <= self.new_arrivals_days)
        )
        segments[new_arrival_mask] = 'New Arrival'

        # Clearance: Old inventory with no recent sales (configurable days threshold)
        clearance_mask = (
            (data['days_since_last_sale'] > self.clearance_days) &
            (data['units_sold_90d'] == 0) &
            (data['total_active_qty'] > 0)
        )
        segments[clearance_mask] = 'Clearance'

        # One-Time Purchase
        one_time_mask = (
            (data['receive_count'] == 1) &
            (~new_arrival_mask) &
            (~clearance_mask)
        )
        segments[one_time_mask] = 'One-Time'

        # Non-Core Repeat
        non_core_mask = (
            (data['receive_count'] >= 2) &
            (data['receive_count'] < 6) &
            (~best_seller_mask) &
            (~new_arrival_mask)
        )
        segments[non_core_mask] = 'Non-Core Repeat'

        return segments

    def train(self, days_back: int = 90) -> Dict:
        """
        Train the segmentation model.

        Uses:
        1. K-Means clustering to discover natural groupings
        2. Random Forest to classify into business segments
        """
        print(f"Training segmentation model with {days_back} days of data...")

        # Build filter clauses
        where_clauses = self._build_where_clauses()
        store_sum = self._build_store_sum_clause()

        # Build WHERE clause for item_list
        base_where = "i.style_number IS NOT NULL"
        if where_clauses:
            base_where += " AND " + " AND ".join(where_clauses)

        # Get sales period from filters or use days_back
        sales_period_days = self.filters.get('sales_period_days', days_back)

        print(f"Applied filters: {len(where_clauses)} conditions")
        print(f"Store sum clause: {store_sum[:50]}...")
        print(f"Sales period: {sales_period_days} days")

        # Build receiving history CTE if enabled
        receiving_cte = ""
        receiving_join = ""
        receiving_columns = ""

        if self.filters.get('includeReceivingHistory', False):
            receiving_days = self.filters.get('receivingHistoryDays', 180)
            receiving_cte = f"""
            ,
            receiving_metrics AS (
                SELECT
                    i.style_number,
                    COUNT(DISTINCT rv.voucher_number) as receiving_frequency,
                    COUNT(rl.id) as total_receives,
                    SUM(rl.qty) as total_qty_received,
                    AVG(rl.cost::numeric) as avg_receiving_cost,
                    STDDEV(rl.cost::numeric) as cost_volatility,
                    MAX(rv.date) as last_receiving_date,
                    AVG(CASE WHEN rv.type = 'Reversal' THEN 1 ELSE 0 END) as reversal_rate,
                    CASE
                        WHEN COUNT(DISTINCT rv.date) > 1
                        THEN EXTRACT(EPOCH FROM (MAX(rv.date) - MIN(rv.date))) / 86400.0 / NULLIF(COUNT(DISTINCT rv.date) - 1, 0)
                        ELSE NULL
                    END as avg_days_between_receives
                FROM item_list i
                JOIN receiving_lines rl ON i.item_number = rl.item_number
                JOIN receiving_vouchers rv ON rl.voucher_id = rv.id
                WHERE i.style_number IS NOT NULL
                AND rv.date >= CURRENT_DATE - INTERVAL '{receiving_days} days'
                GROUP BY i.style_number
            )
            """
            receiving_join = "LEFT JOIN receiving_metrics rm ON sm.style_number = rm.style_number"
            receiving_columns = """,
                COALESCE(rm.receiving_frequency, 0) as receiving_frequency,
                COALESCE(rm.total_receives, 0) as total_receives,
                COALESCE(rm.total_qty_received, 0) as total_qty_received,
                COALESCE(rm.avg_receiving_cost, 0) as avg_receiving_cost,
                COALESCE(rm.cost_volatility, 0) as cost_volatility,
                COALESCE(rm.reversal_rate, 0) as reversal_rate,
                COALESCE(rm.avg_days_between_receives, 999) as avg_days_between_receives,
                CASE
                    WHEN rm.last_receiving_date IS NOT NULL
                    THEN CURRENT_DATE - rm.last_receiving_date
                    ELSE NULL
                END as days_since_last_receiving"""

        # Fetch training data from database
        query = f"""
            WITH style_metrics AS (
                SELECT
                    i.style_number,
                    MAX(i.item_name) as item_name,
                    MAX(i.category) as category,
                    MAX(i.vendor_name) as vendor_name,
                    MAX(i.gender) as gender,
                    SUM({store_sum}) as total_active_qty,
                    AVG(i.order_cost::numeric) as avg_order_cost,
                    AVG(i.selling_price::numeric) as avg_selling_price,
                    AVG(CASE
                        WHEN i.selling_price::numeric > 0
                        THEN ((i.selling_price::numeric - i.order_cost::numeric) / i.selling_price::numeric * 100)
                        ELSE 0
                    END) as avg_margin_percent,
                    SUM(({store_sum}) * i.order_cost::numeric) as inventory_value,
                    MAX(i.last_rcvd) as last_received,
                    COUNT(DISTINCT i.item_number) as receive_count
                FROM item_list i
                WHERE {base_where}
                GROUP BY i.style_number
            ),
            style_sales AS (
                SELECT
                    i.style_number,
                    COUNT(*) FILTER (WHERE s.date >= CURRENT_DATE - INTERVAL '30 days') as units_sold_30d,
                    COUNT(*) FILTER (WHERE s.date >= CURRENT_DATE - INTERVAL '90 days') as units_sold_90d,
                    MAX(s.date) as last_sale_date
                FROM sales_transactions s
                JOIN item_list i ON s.sku = i.item_number
                WHERE i.style_number IS NOT NULL
                AND s.date >= CURRENT_DATE - INTERVAL '{sales_period_days} days'
                GROUP BY i.style_number
            ){receiving_cte}
            SELECT
                sm.*,
                COALESCE(ss.units_sold_30d, 0) as units_sold_30d,
                COALESCE(ss.units_sold_90d, 0) as units_sold_90d,
                COALESCE(ss.last_sale_date, NULL) as last_sale_date,
                CASE
                    WHEN ss.last_sale_date IS NOT NULL
                    THEN CURRENT_DATE - ss.last_sale_date
                    ELSE NULL
                END as days_since_last_sale,
                CASE
                    WHEN sm.last_received IS NOT NULL
                    THEN CURRENT_DATE - sm.last_received
                    ELSE NULL
                END as days_since_last_receive,
                CASE
                    WHEN sm.total_active_qty > 0
                    THEN COALESCE(ss.units_sold_30d, 0)::numeric / 30.0
                    ELSE 0
                END as sales_velocity,
                CASE
                    WHEN sm.avg_selling_price > 0
                    THEN sm.avg_selling_price - sm.avg_order_cost
                    ELSE 0
                END as margin_per_unit{receiving_columns}
            FROM style_metrics sm
            LEFT JOIN style_sales ss ON sm.style_number = ss.style_number
            {receiving_join}
            WHERE sm.total_active_qty > 0
        """.format(days_back=days_back)

        data = db.execute_query(query)

        if data.empty or len(data) < 10:
            raise ValueError("Insufficient data for training (need at least 10 products)")

        print(f"Loaded {len(data)} products for training")

        # Extract features
        X = self.extract_features(data)

        # Get rule-based segment labels
        y = self.assign_rule_based_segments(data)

        print(f"Segment distribution:\n{y.value_counts()}")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 1. K-Means clustering for pattern discovery
        optimal_k = min(8, len(data) // 20)  # Don't create too many clusters
        self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = self.cluster_model.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, clusters)
        print(f"K-Means clustering: {optimal_k} clusters, silhouette score: {silhouette:.3f}")

        # Add cluster as a feature
        X['cluster'] = clusters

        # 2. Train Random Forest classifier
        # Filter out Unknown segments for training
        mask = y != 'Unknown'
        X_train_full = X[mask]
        y_train_full = y[mask]

        if len(X_train_full) < 10:
            raise ValueError("Insufficient labeled data for training")

        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced segments
        )

        self.classification_model.fit(X_train, y_train)

        # Evaluate
        train_score = self.classification_model.score(X_train, y_train)
        test_score = self.classification_model.score(X_test, y_test)

        predictions = self.classification_model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)

        self.metrics = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'silhouette_score': float(silhouette),
            'n_clusters': optimal_k,
            'training_samples': len(X_train_full),
            'test_samples': len(X_test),
            'segment_distribution': y.value_counts().to_dict(),
            'classification_report': report,
        }

        # Set model version
        self.training_date = datetime.now()
        self.model_version = f"v{self.training_date.strftime('%Y%m%d_%H%M%S')}"

        print(f"✓ Segmentation model trained successfully")
        print(f"  Train accuracy: {train_score:.3f}")
        print(f"  Test accuracy: {test_score:.3f}")
        print(f"  Model version: {self.model_version}")

        return self.metrics

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict product segments using trained model.

        Returns enriched data with ML-predicted segments.
        """
        if self.classification_model is None or self.cluster_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features
        X = self.extract_features(data)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict clusters
        clusters = self.cluster_model.predict(X_scaled)
        X['cluster'] = clusters

        # Predict segments
        segments = self.classification_model.predict(X)
        probabilities = self.classification_model.predict_proba(X)

        # Get confidence scores (max probability for predicted class)
        confidence_scores = np.max(probabilities, axis=1)

        # Add predictions to data
        result = data.copy()
        result['ml_segment'] = segments
        result['ml_confidence'] = confidence_scores
        result['cluster_id'] = clusters

        return result

    def save_model(self, filename: Optional[str] = None):
        """Save trained model to disk."""
        if filename is None:
            filename = f"segmentation_{self.model_version}.pkl"

        filepath = os.path.join(settings.model_cache_dir, filename)

        model_data = {
            'cluster_model': self.cluster_model,
            'classification_model': self.classification_model,
            'scaler': self.scaler,
            'model_version': self.model_version,
            'training_date': self.training_date,
            'metrics': self.metrics,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load_latest(cls) -> 'SegmentationPredictor':
        """Load the most recently trained model."""
        model_files = [
            f for f in os.listdir(settings.model_cache_dir)
            if f.startswith('segmentation_') and f.endswith('.pkl')
        ]

        if not model_files:
            raise FileNotFoundError("No trained segmentation model found")

        # Sort by filename (contains timestamp)
        latest_file = sorted(model_files)[-1]
        filepath = os.path.join(settings.model_cache_dir, latest_file)

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls()
        predictor.cluster_model = model_data['cluster_model']
        predictor.classification_model = model_data['classification_model']
        predictor.scaler = model_data['scaler']
        predictor.model_version = model_data['model_version']
        predictor.training_date = model_data['training_date']
        predictor.metrics = model_data['metrics']

        return predictor
