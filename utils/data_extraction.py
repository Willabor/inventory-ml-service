"""Data extraction queries for ML training and prediction."""
from utils.database import db
import pandas as pd
from datetime import datetime, timedelta


def extract_transfer_training_data(days_back: int = 90) -> pd.DataFrame:
    """
    Extract training data for transfer prediction model.

    Creates examples of past transfer scenarios with outcome labels.
    For each style-from_store-to_store combination, we look at:
    - Historical stock levels
    - Sales velocities
    - Whether a transfer would have been successful

    Args:
        days_back: How many days of historical data to use

    Returns:
        DataFrame with features (X) and target (y) for training
    """

    query = """
    WITH store_sales_velocity AS (
        -- Calculate 30-day sales velocity per style per store
        SELECT
            il.style_number,
            st.store,
            COUNT(st.id) as sales_count_30d,
            COUNT(st.id)::numeric / 30.0 as avg_daily_sales,
            MAX(st.date) as last_sale_date,
            AVG(CAST(st.price AS NUMERIC)) as avg_sale_price
        FROM sales_transactions st
        INNER JOIN item_list il ON st.sku = il.item_number
        WHERE
            st.date >= CURRENT_DATE - INTERVAL '%(days_back)s days'
            AND il.style_number IS NOT NULL
            AND il.style_number != ''
            AND st.store IN ('GM', 'HM', 'NM', 'LM')
        GROUP BY il.style_number, st.store
    ),

    store_inventory AS (
        -- Get average stock levels per style per store
        SELECT
            style_number,
            AVG(COALESCE(gm_qty, 0)) as avg_gm_qty,
            AVG(COALESCE(hm_qty, 0)) as avg_hm_qty,
            AVG(COALESCE(nm_qty, 0)) as avg_nm_qty,
            AVG(COALESCE(lm_qty, 0)) as avg_lm_qty,
            AVG(CAST(COALESCE(order_cost, '0') AS NUMERIC)) as avg_order_cost,
            AVG(CAST(COALESCE(selling_price, '0') AS NUMERIC)) as avg_selling_price,
            MAX(item_name) as item_name,
            MAX(category) as category,
            MAX(vendor_name) as vendor_name
        FROM item_list
        WHERE
            style_number IS NOT NULL
            AND style_number != ''
        GROUP BY style_number
    )

    SELECT
        si.style_number,
        si.item_name,
        si.category,
        si.vendor_name,

        -- From store metrics (GM as example, will repeat for all stores)
        CASE
            WHEN from_stores.store = 'GM' THEN si.avg_gm_qty
            WHEN from_stores.store = 'HM' THEN si.avg_hm_qty
            WHEN from_stores.store = 'NM' THEN si.avg_nm_qty
            WHEN from_stores.store = 'LM' THEN si.avg_lm_qty
        END as from_store_qty,
        COALESCE(from_store.avg_daily_sales, 0) as from_store_daily_sales,

        -- To store metrics
        CASE
            WHEN to_stores.store = 'GM' THEN si.avg_gm_qty
            WHEN to_stores.store = 'HM' THEN si.avg_hm_qty
            WHEN to_stores.store = 'NM' THEN si.avg_nm_qty
            WHEN to_stores.store = 'LM' THEN si.avg_lm_qty
        END as to_store_qty,
        COALESCE(to_store.avg_daily_sales, 0) as to_store_daily_sales,

        -- Store identifiers
        from_stores.store as from_store,
        to_stores.store as to_store,

        -- Pricing
        si.avg_order_cost,
        si.avg_selling_price,
        CASE
            WHEN si.avg_selling_price > 0 THEN
                ((si.avg_selling_price - si.avg_order_cost) / si.avg_selling_price) * 100
            ELSE 0
        END as margin_percent,

        -- Target: Would transfer have been successful?
        -- Success = TO store sales velocity > FROM store sales velocity AND TO store actually sold items
        CASE
            WHEN COALESCE(to_store.avg_daily_sales, 0) > COALESCE(from_store.avg_daily_sales, 0) * 1.5
                 AND COALESCE(to_store.sales_count_30d, 0) >= 3  -- At least 3 sales in 30 days
            THEN 1
            ELSE 0
        END as transfer_success

    FROM store_inventory si
    CROSS JOIN (SELECT 'GM' as store UNION SELECT 'HM' UNION SELECT 'NM' UNION SELECT 'LM') from_stores
    CROSS JOIN (SELECT 'GM' as store UNION SELECT 'HM' UNION SELECT 'NM' UNION SELECT 'LM') to_stores
    LEFT JOIN store_sales_velocity from_store ON from_store.style_number = si.style_number
        AND from_store.store = from_stores.store
    LEFT JOIN store_sales_velocity to_store ON to_store.style_number = si.style_number
        AND to_store.store = to_stores.store

    WHERE
        from_stores.store != to_stores.store  -- Don't transfer to same store
        AND si.avg_gm_qty + si.avg_hm_qty + si.avg_nm_qty + si.avg_lm_qty > 5  -- Has some inventory

    ORDER BY si.style_number, from_stores.store, to_stores.store
    """

    df = db.execute_query(query, {'days_back': days_back})
    return df


def extract_transfer_prediction_candidates(limit: int = 100) -> pd.DataFrame:
    """
    Extract current transfer candidates for prediction.

    Gets the latest inventory and sales data to identify potential transfers.

    Args:
        limit: Maximum number of candidates to return

    Returns:
        DataFrame with current transfer candidates
    """

    query = """
    WITH latest_inventory AS (
        -- Get most recent inventory snapshot per style
        SELECT
            style_number,
            item_name,
            category,
            vendor_name,
            SUM(COALESCE(gm_qty, 0)) as gm_qty,
            SUM(COALESCE(hm_qty, 0)) as hm_qty,
            SUM(COALESCE(nm_qty, 0)) as nm_qty,
            SUM(COALESCE(lm_qty, 0)) as lm_qty,
            AVG(CAST(COALESCE(order_cost, '0') AS NUMERIC)) as avg_order_cost,
            AVG(CAST(COALESCE(selling_price, '0') AS NUMERIC)) as avg_selling_price
        FROM item_list
        WHERE
            style_number IS NOT NULL
            AND style_number != ''
        GROUP BY style_number, item_name, category, vendor_name
    ),

    recent_sales AS (
        -- Sales in last 30 days per style per store
        SELECT
            il.style_number,
            st.store,
            COUNT(st.id)::numeric / 30.0 as avg_daily_sales
        FROM sales_transactions st
        INNER JOIN item_list il ON st.sku = il.item_number
        WHERE
            st.date >= CURRENT_DATE - INTERVAL '30 days'
            AND il.style_number IS NOT NULL
            AND st.store IN ('GM', 'HM', 'NM', 'LM')
        GROUP BY il.style_number, st.store
    )

    SELECT
        li.style_number,
        li.item_name,
        li.category,
        li.vendor_name,

        from_store.store as from_store,
        to_store.store as to_store,

        -- Stock levels
        CASE
            WHEN from_store.store = 'GM' THEN li.gm_qty
            WHEN from_store.store = 'HM' THEN li.hm_qty
            WHEN from_store.store = 'NM' THEN li.nm_qty
            WHEN from_store.store = 'LM' THEN li.lm_qty
        END as from_store_qty,

        CASE
            WHEN to_store.store = 'GM' THEN li.gm_qty
            WHEN to_store.store = 'HM' THEN li.hm_qty
            WHEN to_store.store = 'NM' THEN li.nm_qty
            WHEN to_store.store = 'LM' THEN li.lm_qty
        END as to_store_qty,

        -- Sales velocities
        COALESCE(from_sales.avg_daily_sales, 0) as from_store_daily_sales,
        COALESCE(to_sales.avg_daily_sales, 0) as to_store_daily_sales,

        -- Pricing
        li.avg_order_cost,
        li.avg_selling_price,
        CASE
            WHEN li.avg_selling_price > 0 THEN
                ((li.avg_selling_price - li.avg_order_cost) / li.avg_selling_price) * 100
            ELSE 0
        END as margin_percent

    FROM latest_inventory li
    CROSS JOIN (SELECT 'GM' as store UNION SELECT 'HM' UNION SELECT 'NM' UNION SELECT 'LM') from_store
    CROSS JOIN (SELECT 'GM' as store UNION SELECT 'HM' UNION SELECT 'NM' UNION SELECT 'LM') to_store
    LEFT JOIN recent_sales from_sales ON from_sales.style_number = li.style_number
        AND from_sales.store = from_store.store
    LEFT JOIN recent_sales to_sales ON to_sales.style_number = li.style_number
        AND to_sales.store = to_store.store

    WHERE
        from_store.store != to_store.store
        -- Basic transfer criteria
        AND CASE
            WHEN from_store.store = 'GM' THEN li.gm_qty
            WHEN from_store.store = 'HM' THEN li.hm_qty
            WHEN from_store.store = 'NM' THEN li.nm_qty
            WHEN from_store.store = 'LM' THEN li.lm_qty
        END > 5  -- FROM store has excess
        AND COALESCE(to_sales.avg_daily_sales, 0) > COALESCE(from_sales.avg_daily_sales, 0)  -- TO sells faster

    ORDER BY
        COALESCE(to_sales.avg_daily_sales, 0) - COALESCE(from_sales.avg_daily_sales, 0) DESC

    LIMIT %(limit)s
    """

    df = db.execute_query(query, {'limit': limit})
    return df
