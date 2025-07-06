from dagster import AssetOut, asset, multi_asset
import pandas as pd

group_name = "get_features"

@multi_asset(group_name=group_name, 
       outs={"df_input_preprocessed": AssetOut()},)
def df_input_preprocessed(core_data, usage_info, customer_interactions): 
    features_df = core_data.copy()

    # Extract features
    features_df = features_df.merge(
        get_usage_features(usage_info),
        on='rating_account_id',
        how='left'
    ).merge(
        get_interaction_type_features(customer_interactions),
        on='customer_id',
        how='left'
    )
    return features_df

def get_usage_features(usage_info):

    usage_features = (
    usage_info
    .groupby('rating_account_id', as_index=False)
    .agg({
        'has_used_roaming': 'max',
        'used_gb': 'sum'
    })
    .rename(columns={
        'has_used_roaming': 'has_used_roaming_4M',
        'used_gb': 'used_gb_sum_4M'
    })
)

    return usage_features

def get_interaction_type_features(customer_interactions):
    # Aggregate by customer_id and type_subtype
    agg_df = customer_interactions.groupby(['customer_id', 'type_subtype']).agg(
        n=('n', 'sum'),
        days_since_last=('days_since_last', 'min')
    ).reset_index()

    # Pivot interaction types into columns
    pivot_n = agg_df.pivot(index='customer_id', columns='type_subtype', values='n')
    pivot_days = agg_df.pivot(index='customer_id', columns='type_subtype', values='days_since_last')

    # Rename columns to avoid name collisions
    pivot_n.columns = [f'{col}_n' for col in pivot_n.columns]
    pivot_days.columns = [f'days_since_last_{col}' for col in pivot_days.columns]

    # Combine all features
    interaction_features = pd.concat([pivot_n, pivot_days], axis=1).reset_index()

    # Binary indicator for whether a customer asked about a given type
    for col in pivot_n.columns:
        indicator_col = f'asked_{col[:-2]}'  # Remove '_n' from name
        interaction_features[indicator_col] = interaction_features[col].notnull().astype(int)

    return interaction_features



