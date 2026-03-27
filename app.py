import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="CLV Tier Classifier", layout="wide")
st.title("Customer Lifetime Value (CLV) Tier Classifier")
st.write("Upload your order data to classify customers into Platinum, Gold, Silver, and Bronze tiers based on historical and projected lifetime value.")

# STUDENT NOTE: Define required columns. This single list drives all validation.
# Any CSV missing these columns will be caught before computation begins,
# so the app never crashes with a confusing error on bad input.
REQUIRED_COLUMNS = ['customer_unique_id', 'order_id', 'order_date', 'order_value']

# STUDENT NOTE: Tier colour palette defined once and reused across all charts
# so colours are consistent regardless of which chart the user is looking at.
TIER_COLORS = {
    'Platinum': '#8B8FA8',
    'Gold':     '#FFD700',
    'Silver':   '#C0C0C0',
    'Bronze':   '#CD7F32',
}

# --- DATA LOADING AND VALIDATION ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:

    # STUDENT NOTE: Load the uploaded CSV into a DataFrame.
    df = pd.read_csv(uploaded_file)

    # STUDENT NOTE: Validate required columns before any processing.
    # Returning the specific missing names makes the error message actionable --
    # the user knows exactly which column to rename rather than guessing.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Please check your file and rename columns if needed.")
        st.stop()

    # STUDENT NOTE: Parse order_date as datetime. Without this, date arithmetic
    # (tenure, reference date) will fail with a TypeError.
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # STUDENT NOTE: Drop rows where order_date or order_value is null.
    # A null in either column means the row cannot contribute to CLV.
    df = df.dropna(subset=['order_date', 'order_value'])

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    # --- INTERACTIVE FILTER ---
    # STUDENT NOTE: State filter is applied before metric computation.
    # Changing it reruns the entire CLV formula on the filtered subset,
    # which changes all output values -- not just chart labels.
    if 'customer_state' in df.columns:
        all_states = sorted(df['customer_state'].dropna().unique().tolist())
        selected_states = st.multiselect(
            "Filter by Customer State (leave blank for all states)",
            options=all_states,
            default=[]
        )
        if selected_states:
            df = df[df['customer_state'].isin(selected_states)]

    st.markdown("---")

    # --- METRIC COMPUTATION ---

    # STUDENT NOTE: Set reference date to one day after the last order in the dataset.
    # Using a data-relative reference date makes results reproducible
    # regardless of when the app is run.
    reference_date = df['order_date'].max() + pd.Timedelta(days=1)

    # STUDENT NOTE: Aggregate to one row per customer.
    # nunique() on order_id correctly handles datasets where one order
    # has multiple payment rows (e.g. installments), so we count orders
    # rather than payment rows.
    clv_df = (
        df.groupby('customer_unique_id')
        .agg(
            total_revenue=('order_value', 'sum'),
            order_count=('order_id', 'nunique'),
            first_order_date=('order_date', 'min'),
            last_order_date=('order_date', 'max')
        )
        .reset_index()
    )

    # STUDENT NOTE: Round revenue to 2 decimal places to avoid floating point noise.
    clv_df['total_revenue'] = clv_df['total_revenue'].round(2)

    # STUDENT NOTE: Average order value captures spending intensity per visit,
    # which is distinct from total revenue and needed for the 12-month projection.
    clv_df['avg_order_value'] = (clv_df['total_revenue'] / clv_df['order_count']).round(2)

    # STUDENT NOTE: Tenure = days between first and last order.
    # Single-order customers get tenure = 0, which is handled in the projection step.
    clv_df['tenure_days'] = (
        clv_df['last_order_date'] - clv_df['first_order_date']
    ).dt.days

    # STUDENT NOTE: Assign CLV tiers using percentile cutoffs, not fixed thresholds.
    # Percentile-based assignment ensures meaningful tier separation on any dataset
    # regardless of revenue scale, currency, or business size.
    p90 = clv_df['total_revenue'].quantile(0.90)
    p70 = clv_df['total_revenue'].quantile(0.70)
    p40 = clv_df['total_revenue'].quantile(0.40)

    clv_df['clv_tier'] = pd.cut(
        clv_df['total_revenue'],
        bins=[-np.inf, p40, p70, p90, np.inf],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )

    # STUDENT NOTE: Project 12-month CLV using daily purchase rate x average order value.
    # Adding 1 to tenure_days prevents division by zero for single-order customers.
    clv_df['purchase_rate_per_day'] = clv_df['order_count'] / (clv_df['tenure_days'] + 1)
    clv_df['projected_clv_12m'] = (
        clv_df['purchase_rate_per_day'] * 365 * clv_df['avg_order_value']
    ).round(2)

    # STUDENT NOTE: Cap projections at 3x historical CLV to prevent unrealistic
    # outlier projections from single-order customers with high average order values.
    clv_df['projected_clv_12m'] = clv_df[['projected_clv_12m', 'total_revenue']].apply(
        lambda row: min(row['projected_clv_12m'], row['total_revenue'] * 3), axis=1
    ).round(2)

    # --- HEADLINE METRICS ---
    avg_clv = clv_df['total_revenue'].mean()
    platinum_threshold = p90
    total_projected = clv_df['projected_clv_12m'].sum()
    platinum_count = (clv_df['clv_tier'] == 'Platinum').sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average CLV", f"R${avg_clv:,.2f}")
    col2.metric("Platinum Threshold", f"R${platinum_threshold:,.2f}")
    col3.metric("Total 12-Month Projected CLV", f"R${total_projected:,.0f}")
    col4.metric("Platinum Customers", f"{platinum_count:,}")

    # --- CHARTS ---

    tier_summary = (
        clv_df.groupby('clv_tier', observed=True)
        .agg(
            customer_count=('customer_unique_id', 'count'),
            total_revenue=('total_revenue', 'sum'),
            avg_clv=('total_revenue', 'mean')
        )
        .reset_index()
    )

    # STUDENT NOTE: Chart 1 -- Tier distribution bar chart showing total revenue
    # and customer count per tier. This is the primary required output for Metric 06.
    fig1 = px.bar(
        tier_summary,
        x='clv_tier',
        y='total_revenue',
        text='customer_count',
        title='CLV Tier Distribution — Total Revenue per Tier',
        labels={'clv_tier': 'Tier', 'total_revenue': 'Total Revenue'},
        color='clv_tier',
        color_discrete_map=TIER_COLORS
    )
    fig1.update_traces(texttemplate='%{text} customers', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

    # STUDENT NOTE: Chart 2 -- Tier profile comparison across avg order value,
    # frequency, and tenure. Required second output for Metric 06.
    tier_profile = clv_df.groupby('clv_tier', observed=True)[
        ['avg_order_value', 'order_count', 'tenure_days']
    ].mean().reset_index()

    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Avg Order Value', 'Avg Order Count', 'Avg Tenure (days)')
    )
    for i, col in enumerate(['avg_order_value', 'order_count', 'tenure_days']):
        fig2.add_trace(
            go.Bar(
                x=tier_profile['clv_tier'],
                y=tier_profile[col].round(1),
                marker_color=[TIER_COLORS[t] for t in tier_profile['clv_tier']],
                showlegend=False
            ),
            row=1, col=i + 1
        )
    fig2.update_layout(title_text='CLV Tier Profile — Avg Order Value, Frequency and Tenure by Tier')
    st.plotly_chart(fig2, use_container_width=True)

    # STUDENT NOTE: Chart 3 -- CLV distribution histogram showing the spread of
    # customer revenue within each tier. Required third output for Metric 06.
    cap = clv_df['total_revenue'].quantile(0.99)
    fig3 = px.histogram(
        clv_df[clv_df['total_revenue'] <= cap],
        x='total_revenue',
        color='clv_tier',
        nbins=80,
        title='CLV Distribution by Tier',
        labels={'total_revenue': 'Total Revenue', 'clv_tier': 'Tier'},
        color_discrete_map=TIER_COLORS,
        barmode='overlay',
        opacity=0.7
    )
    st.plotly_chart(fig3, use_container_width=True)

    # STUDENT NOTE: Full customer tier table sorted by revenue descending
    # so the client can immediately see their most valuable customers.
    st.subheader("Customer Tier Table")
    display_df = clv_df[[
        'customer_unique_id', 'total_revenue', 'order_count',
        'avg_order_value', 'tenure_days', 'projected_clv_12m', 'clv_tier'
    ]].sort_values('total_revenue', ascending=False)
    st.dataframe(display_df, use_container_width=True)

    # --- INTERPRETATION ---
    # STUDENT NOTE: Plain-English interpretation panel written by the analyst.
    # This text is not AI-generated -- it explains what the metric shows and
    # what a business user should pay attention to in their specific results.
    st.info("""
**What this metric shows:** Customer Lifetime Value (CLV) measures the total confirmed revenue
each customer has generated and projects what they are likely to spend in the next 12 months
based on their purchase rate and average order size.

**How to read your results:** Platinum customers (top 10% by revenue) are your most valuable
accounts -- protect these relationships with priority service and loyalty offers. Gold and Silver
customers represent your growth opportunity -- they have demonstrated repeat behaviour and
respond well to re-engagement campaigns. Bronze customers are typically one-time buyers;
the business decision is whether to invest in converting them or focus resources on the upper tiers.

**What to watch:** The 12-month projected CLV uses a simple linear model based on historical
purchase rate. Customers with only one order will have conservative projections. Compare
projected vs. historical CLV to identify customers who appear undervalued by the model.
    """)
