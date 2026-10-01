import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TechPulse | Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL THEME
# ============================================================

sns.set_theme(style="whitegrid")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #101828;
}

[data-testid="stSidebar"] * {
    color: #F2F4F7 !important;
}

/* Hero */

.hero {
    background: linear-gradient(
        120deg,
        #101828 0%,
        #1D3557 55%,
        #2563EB 100%
    );

    padding: 30px 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;

    box-shadow:
        0 15px 35px rgba(16,24,40,0.12);
}

.hero h1 {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    color: #D9E5F7;
    font-size: 14px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #101828;
    margin-top: 10px;
    margin-bottom: 18px;
}

/* Metrics */

div[data-testid="stMetric"] {

    background: white;

    border: 1px solid #E4E7EC;

    padding: 17px;

    border-radius: 15px;

    box-shadow:
        0 4px 15px rgba(16,24,40,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #667085;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {

    color: #101828;

    font-family: 'Manrope', sans-serif;

    font-weight: 800;
}

/* Tables */

div[data-testid="stDataFrame"] {

    border: 1px solid #E4E7EC;

    border-radius: 12px;

    overflow: hidden;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {

    border-radius: 10px;

    font-weight: 700;

}

/* Pills */

.pill {

    display: inline-block;

    padding: 5px 10px;

    border-radius: 20px;

    background: #EAF2FF;

    color: #175CD3;

    font-weight: 700;

    font-size: 12px;

}

/* Information cards */

.info-card {

    background: white;

    padding: 20px;

    border-radius: 15px;

    border: 1px solid #E4E7EC;

    box-shadow:
        0 4px 15px rgba(16,24,40,0.04);

}

.small-text {

    color: #667085;

    font-size: 13px;

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCTIONS
# ============================================================

def money(value):

    if pd.isna(value):
        return "₹0"

    return f"₹{value:,.0f}"


def load_dataset():

    st.sidebar.markdown("## ⚙️ Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales CSV",
        type=["csv"]
    )

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    default_file = os.path.join(
        base_dir,
        "mobile_sales_data.csv"
    )

    try:

        if uploaded_file is not None:

            df = pd.read_csv(
                uploaded_file
            )

            filename = uploaded_file.name

        elif os.path.exists(default_file):

            df = pd.read_csv(
                default_file
            )

            filename = "mobile_sales_data.csv"

        else:

            st.warning(
                "Upload your CSV from the sidebar or place "
                "`mobile_sales_data.csv` beside app.py."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

        st.stop()

    return df, filename


def clean_dataset(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required_columns = [

        "Inward Date",
        "Dispatch Date",
        "Price",
        "Quantity Sold",
        "Product",
        "Product Code",
        "Brand",
        "Region"

    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        st.error(
            "The following required columns are missing:\n\n"
            + ", ".join(missing)
        )

        st.write(
            "Columns found in your CSV:"
        )

        st.write(
            df.columns.tolist()
        )

        st.stop()

    # Text cleaning

    text_columns = [

        "Product",
        "Product Code",
        "Brand",
        "Region"

    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # Dates

    df["Inward Date"] = pd.to_datetime(
        df["Inward Date"],
        errors="coerce"
    )

    df["Dispatch Date"] = pd.to_datetime(
        df["Dispatch Date"],
        errors="coerce"
    )

    # Numbers

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

    df["Quantity Sold"] = pd.to_numeric(
        df["Quantity Sold"],
        errors="coerce"
    )

    # Revenue

    df["Revenue"] = (
        df["Price"] *
        df["Quantity Sold"]
    )

    # Dispatch time

    df["Dispatch Lead Time (Days)"] = (

        df["Dispatch Date"] -
        df["Inward Date"]

    ).dt.days

    # Month

    df["Inward Month"] = (

        df["Inward Date"]
        .dt.to_period("M")
        .astype(str)

    )

    # Year

    df["Year"] = (
        df["Inward Date"]
        .dt.year
    )

    # Product category

    def detect_type(product):

        product = str(product).lower()

        if (
            "laptop" in product
            or "notebook" in product
            or "macbook" in product
        ):

            return "Laptop"

        if (
            "mobile" in product
            or "phone" in product
            or "smartphone" in product
            or "iphone" in product
            or "galaxy" in product
        ):

            return "Mobile"

        return "Other"

    df["Product Type"] = (
        df["Product"]
        .apply(detect_type)
    )

    # Valid data

    analysis_data = df.dropna(
        subset=[
            "Price",
            "Quantity Sold",
            "Inward Date",
            "Dispatch Date"
        ]
    ).copy()

    analysis_data = analysis_data[
        (analysis_data["Price"] >= 0)
        &
        (analysis_data["Quantity Sold"] >= 0)
    ]

    return analysis_data


def remove_chart_borders(ax):

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(True)


def show_kpis(data):

    total_revenue = (
        data["Revenue"].sum()
    )

    total_units = (
        data["Quantity Sold"].sum()
    )

    avg_price = (
        data["Price"].mean()
    )

    total_products = (
        data["Product"].nunique()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Total Revenue",
        money(total_revenue)
    )

    c2.metric(
        "📦 Units Sold",
        f"{total_units:,.0f}"
    )

    c3.metric(
        "🏷️ Average Price",
        money(avg_price)
    )

    c4.metric(
        "📱 Products",
        f"{total_products:,}"
    )


# ============================================================
# LOAD DATA
# ============================================================

raw_df, dataset_name = load_dataset()

data = clean_dataset(
    raw_df
)

if data.empty:

    st.error(
        "No valid records remain after cleaning."
    )

    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "## 📍 Dashboard"
)

page = st.sidebar.radio(

    "Select page",

    [

        "🏠 Executive Overview",

        "🔎 Product Explorer",

        "📈 Sales Intelligence",

        "🚚 Operations",

        "🤖 ML Price Predictor",

        "🧹 Data Quality"

    ]

)


# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "## 🎛️ Global Filters"
)

product_types = sorted(
    data["Product Type"]
    .unique()
    .tolist()
)

selected_types = st.sidebar.multiselect(

    "Product Type",

    product_types,

    default=product_types

)


brands = sorted(
    data["Brand"]
    .unique()
    .tolist()
)

selected_brands = st.sidebar.multiselect(

    "Brand",

    brands,

    default=brands

)


regions = sorted(
    data["Region"]
    .unique()
    .tolist()
)

selected_regions = st.sidebar.multiselect(

    "Region",

    regions,

    default=regions

)


min_price = float(
    data["Price"].min()
)

max_price = float(
    data["Price"].max()
)

if min_price == max_price:

    selected_price = (
        min_price,
        max_price
    )

else:

    selected_price = st.sidebar.slider(

        "💰 Price Range",

        min_value=min_price,

        max_value=max_price,

        value=(
            min_price,
            max_price
        ),

        step=max(
            (max_price - min_price) / 100,
            1
        )

    )


# Apply filters

filtered_data = data[

    data["Product Type"].isin(
        selected_types
    )

    &

    data["Brand"].isin(
        selected_brands
    )

    &

    data["Region"].isin(
        selected_regions
    )

    &

    data["Price"].between(
        selected_price[0],
        selected_price[1]
    )

].copy()


if filtered_data.empty:

    st.warning(
        "No records match the selected filters."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""

<div class="hero">

<span class="pill">
SALES INTELLIGENCE
</span>

<h1>
📊 TechPulse Analytics
</h1>

<p>
Premium Mobile & Laptop Sales Analytics,
Performance Intelligence and Machine Learning.
</p>

</div>

""", unsafe_allow_html=True)


st.caption(

    f"Dataset: {dataset_name}  |  "
    f"Showing {len(filtered_data):,} of "
    f"{len(data):,} valid records"

)


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.markdown(
        '<div class="section-title">'
        'Executive Business Overview'
        '</div>',
        unsafe_allow_html=True
    )

    show_kpis(
        filtered_data
    )

    st.markdown("")

    # Revenue trend + category mix

    left, right = st.columns(
        [1.5, 1]
    )

    # Monthly revenue

    with left:

        st.markdown(
            "#### 📈 Revenue Trend"
        )

        monthly = (

            filtered_data
            .groupby(
                "Inward Month",
                as_index=False
            )
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                )
            )
            .sort_values(
                "Inward Month"
            )

        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        sns.lineplot(

            data=monthly,

            x="Inward Month",

            y="Revenue",

            marker="o",

            linewidth=2.5,

            ax=ax

        )

        ax.set_xlabel(
            "Month"
        )

        ax.set_ylabel(
            "Revenue (₹)"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        remove_chart_borders(
            ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    # Category mix

    with right:

        st.markdown(
            "#### 📱 Category Revenue Mix"
        )

        category = (

            filtered_data
            .groupby(
                "Product Type"
            )["Revenue"]
            .sum()
            .sort_values(
                ascending=False
            )

        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.pie(

            category.values,

            labels=category.index,

            autopct="%1.1f%%",

            startangle=90,

            wedgeprops={
                "width": 0.42
            }

        )

        ax.set_title(
            "Revenue Share"
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


    # Product and region

    left, right = st.columns(
        2
    )


    with left:

        st.markdown(
            "#### 🏆 Top Products"
        )

        top_products = (

            filtered_data
            .groupby(
                "Product",
                as_index=False
            )
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                ),

                Units=(
                    "Quantity Sold",
                    "sum"
                )
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
            .head(10)

        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(

            data=top_products,

            x="Revenue",

            y="Product",

            ax=ax

        )

        ax.set_xlabel(
            "Revenue (₹)"
        )

        ax.set_ylabel("")

        remove_chart_borders(
            ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


    with right:

        st.markdown(
            "#### 🌍 Revenue by Region"
        )

        region = (

            filtered_data
            .groupby(
                "Region",
                as_index=False
            )["Revenue"]
            .sum()
            .sort_values(
                "Revenue",
                ascending=False
            )

        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(

            data=region,

            x="Region",

            y="Revenue",

            ax=ax

        )

        ax.tick_params(
            axis="x",
            rotation=25
        )

        ax.set_ylabel(
            "Revenue (₹)"
        )

        remove_chart_borders(
            ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


    st.markdown(
        "#### 📋 Highest-Value Transactions"
    )

    display_columns = [

        "Inward Date",

        "Product",

        "Brand",

        "Product Type",

        "Region",

        "Price",

        "Quantity Sold",

        "Revenue"

    ]

    st.dataframe(

        filtered_data
        .sort_values(
            "Revenue",
            ascending=False
        )[display_columns]
        .head(20),

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# PAGE 2 — PRODUCT EXPLORER
# ============================================================

elif page == "🔎 Product Explorer":

    st.markdown(
        '<div class="section-title">'
        'Product Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Search products, mobiles, laptops, brands and product codes."
    )

    search = st.text_input(

        "🔎 Search",

        placeholder=(
            "Search iPhone, Samsung, Dell, "
            "laptop, product code..."
        )

    )

    type_filter = st.selectbox(

        "Product category",

        [
            "All",
            "Mobile",
            "Laptop",
            "Other"
        ]

    )

    explorer = filtered_data.copy()


    if search.strip():

        search_text = (
            search.strip()
        )

        search_mask = (

            explorer["Product"]
            .str.contains(
                search_text,
                case=False,
                na=False
            )

            |

            explorer["Brand"]
            .str.contains(
                search_text,
                case=False,
                na=False
            )

            |

            explorer["Product Code"]
            .str.contains(
                search_text,
                case=False,
                na=False
            )

        )

        explorer = explorer[
            search_mask
        ]


    if type_filter != "All":

        explorer = explorer[
            explorer["Product Type"]
            == type_filter
        ]


    if explorer.empty:

        st.info(
            "No matching products found."
        )

    else:

        show_kpis(
            explorer
        )

        product_summary = (

            explorer
            .groupby(
                [
                    "Product",
                    "Brand",
                    "Product Type"
                ],
                as_index=False
            )
            .agg(

                Records=(
                    "Product Code",
                    "count"
                ),

                Units_Sold=(
                    "Quantity Sold",
                    "sum"
                ),

                Average_Price=(
                    "Price",
                    "mean"
                ),

                Revenue=(
                    "Revenue",
                    "sum"
                )

            )
            .sort_values(
                "Revenue",
                ascending=False
            )

        )

        st.markdown(
            "#### 📋 Product Performance"
        )

        st.dataframe(

            product_summary,

            use_container_width=True,

            hide_index=True

        )


        st.markdown(
            "#### 💰 Price Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.boxplot(

            data=explorer,

            x="Product Type",

            y="Price",

            ax=ax

        )

        ax.set_ylabel(
            "Listed Price (₹)"
        )

        remove_chart_borders(
            ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


        st.download_button(

            "⬇️ Download Product Analysis",

            product_summary
            .to_csv(
                index=False
            )
            .encode("utf-8"),

            file_name=(
                "product_analysis.csv"
            ),

            mime="text/csv"

        )


# ============================================================
# PAGE 3 — SALES INTELLIGENCE
# ============================================================

elif page == "📈 Sales Intelligence":

    st.markdown(
        '<div class="section-title">'
        'Sales Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    show_kpis(
        filtered_data
    )

    monthly = (

        filtered_data
        .groupby(
            "Inward Month",
            as_index=False
        )
        .agg(

            Revenue=(
                "Revenue",
                "sum"
            ),

            Units=(
                "Quantity Sold",
                "sum"
            ),

            Average_Price=(
                "Price",
                "mean"
            )

        )
        .sort_values(
            "Inward Month"
        )

    )

    trend_metric = st.selectbox(

        "Select trend",

        [
            "Revenue",
            "Units",
            "Average_Price"
        ]

    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    sns.lineplot(

        data=monthly,

        x="Inward Month",

        y=trend_metric,

        marker="o",

        linewidth=2.5,

        ax=ax

    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    if trend_metric == "Revenue":

        ax.set_ylabel(
            "Revenue (₹)"
        )

    elif trend_metric == "Average_Price":

        ax.set_ylabel(
            "Average Price (₹)"
        )

    else:

        ax.set_ylabel(
            "Units"
        )

    remove_chart_borders(
        ax
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)


    # Brand analysis

    st.markdown(
        "#### 🏷️ Brand Performance"
    )

    brand_summary = (

        filtered_data
        .groupby(
            "Brand",
            as_index=False
        )
        .agg(

            Orders=(
                "Product Code",
                "count"
            ),

            Units_Sold=(
                "Quantity Sold",
                "sum"
            ),

            Revenue=(
                "Revenue",
                "sum"
            ),

            Average_Price=(
                "Price",
                "mean"
            )

        )
        .sort_values(
            "Revenue",
            ascending=False
        )

    )

    st.dataframe(

        brand_summary,

        use_container_width=True,

        hide_index=True

    )


    # Price vs quantity

    st.markdown(
        "#### 🔬 Price vs Quantity"
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    sns.scatterplot(

        data=filtered_data,

        x="Price",

        y="Quantity Sold",

        hue="Product Type",

        alpha=0.65,

        ax=ax

    )

    ax.set_xlabel(
        "Listed Price (₹)"
    )

    ax.set_ylabel(
        "Quantity Sold"
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)


    st.markdown(
        "#### 📅 Monthly Summary"
    )

    st.dataframe(

        monthly,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# PAGE 4 — OPERATIONS
# ============================================================

elif page == "🚚 Operations":

    st.markdown(
        '<div class="section-title">'
        'Dispatch & Operations'
        '</div>',
        unsafe_allow_html=True
    )

    lead_time = filtered_data[
        "Dispatch Lead Time (Days)"
    ]

    negative_count = int(
        (lead_time < 0).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Average Lead Time",

        f"{lead_time.mean():.2f} days"

    )

    c2.metric(

        "Median Lead Time",

        f"{lead_time.median():.2f} days"

    )

    c3.metric(

        "Minimum",

        f"{lead_time.min():.0f} days"

    )

    c4.metric(

        "Negative Lead Times",

        f"{negative_count:,}"

    )


    left, right = st.columns(2)


    with left:

        st.markdown(
            "#### 🚚 Lead Time Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(

            data=filtered_data,

            x="Dispatch Lead Time (Days)",

            bins=25,

            ax=ax

        )

        ax.set_xlabel(
            "Dispatch Lead Time (Days)"
        )

        remove_chart_borders(
            ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


    with right:

        st.markdown(
            "#### 🌍 Average Lead Time by Region"
        )

        region_lead = (

            filtered_data
            .groupby(
                "Region",
                as_index=False
            )[
                "Dispatch Lead Time (Days)"
            ]
            .mean()
            .sort_values(
                "Dispatch Lead Time (Days)"
            )

        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(

            data=region_lead,

            x="Region",

            y="Dispatch Lead Time (Days)",

            ax=ax

        )

        ax.tick_params(
            axis="x",
            rotation=25
        )

        remove_chart_borders(
            ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


    st.markdown(
        "#### 📋 Dispatch Performance"
    )

    dispatch_summary = (

        filtered_data
        .groupby(
            [
                "Region",
                "Product Type"
            ],
            as_index=False
        )
        .agg(

            Orders=(
                "Product Code",
                "count"
            ),

            Average_Lead_Time=(
                "Dispatch Lead Time (Days)",
                "mean"
            ),

            Median_Lead_Time=(
                "Dispatch Lead Time (Days)",
                "median"
            )

        )

        .sort_values(
            "Average_Lead_Time"
        )

    )

    st.dataframe(

        dispatch_summary,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# PAGE 5 — ML PRICE PREDICTOR
# ============================================================

elif page == "🤖 ML Price Predictor":

    st.markdown(
        '<div class="section-title">'
        '🤖 Machine Learning Price Predictor'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        This ML model learns from your historical CSV data.
        It can estimate a price for a new product profile based
        on Product, Brand, Product Type, Region and Quantity Sold.

        The prediction is an experimental data-science estimate,
        not a live market price.
        """
    )


    # --------------------------------------------------------
    # Import ML libraries
    # --------------------------------------------------------

    try:

        from sklearn.compose import ColumnTransformer

        from sklearn.pipeline import Pipeline

        from sklearn.preprocessing import OneHotEncoder

        from sklearn.impute import SimpleImputer

        from sklearn.ensemble import ExtraTreesRegressor

        from sklearn.model_selection import train_test_split

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

    except ImportError:

        st.error(
            "Scikit-learn is not installed."
        )

        st.code(
            "pip install scikit-learn"
        )

        st.stop()


    # --------------------------------------------------------
    # ML DATA
    # --------------------------------------------------------

    ml_data = data.copy()

    features = [

        "Product",

        "Brand",

        "Product Type",

        "Region",

        "Quantity Sold"

    ]

    ml_data = ml_data.dropna(
        subset=features + ["Price"]
    )

    if len(ml_data) < 10:

        st.warning(
            "Your dataset has fewer than 10 usable rows. "
            "The ML results may not be reliable."
        )

    if len(ml_data) < 4:

        st.error(
            "Not enough records for ML training."
        )

        st.stop()


    X = ml_data[
        features
    ]

    y = ml_data[
        "Price"
    ].astype(float)


    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )


    categorical_features = [

        "Product",

        "Brand",

        "Product Type",

        "Region"

    ]

    numeric_features = [

        "Quantity Sold"

    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (

                "categorical",

                Pipeline(

                    steps=[

                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )

                    ]

                ),

                categorical_features

            ),

            (

                "numeric",

                Pipeline(

                    steps=[

                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        )

                    ]

                ),

                numeric_features

            )

        ]

    )


    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (

                "model",

                ExtraTreesRegressor(

                    n_estimators=300,

                    random_state=42,

                    n_jobs=-1

                )

            )

        ]

    )


    # Train

    model.fit(
        X_train,
        y_train
    )


    # Test

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    st.markdown(
        "#### 📊 Model Performance"
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(

        "MAE",

        money(mae)

    )


    m2.metric(

        "RMSE",

        money(rmse)

    )


    m3.metric(

        "R² Score",

        f"{r2:.3f}"

    )


    st.caption(

        """
        MAE and RMSE measure prediction error.
        Lower error is generally better.
        R² shows how much variation in historical price
        is explained by the model. Results depend heavily
        on dataset size and quality.
        """

    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.markdown(
        "#### 🎯 Actual vs Predicted Price"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.scatter(

        y_test,

        predictions,

        alpha=0.7

    )


    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )


    ax.plot(

        [minimum, maximum],

        [minimum, maximum],

        linestyle="--"

    )


    ax.set_xlabel(
        "Actual Price (₹)"
    )

    ax.set_ylabel(
        "Predicted Price (₹)"
    )

    ax.set_title(
        "Actual vs Predicted Price"
    )


    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)


    # --------------------------------------------------------
    # PREDICTION FORM
    # --------------------------------------------------------

    st.markdown(
        "#### 🔮 Predict Price for a New Product"
    )


    col1, col2 = st.columns(2)


    with col1:

        new_product = st.selectbox(

            "📱 Product",

            sorted(
                ml_data[
                    "Product"
                ].unique()
            )

        )


        new_brand = st.selectbox(

            "🏷️ Brand",

            sorted(
                ml_data[
                    "Brand"
                ].unique()
            )

        )


        new_type = st.selectbox(

            "💻 Product Type",

            sorted(
                ml_data[
                    "Product Type"
                ].unique()
            )

        )


    with col2:

        new_region = st.selectbox(

            "🌍 Region",

            sorted(
                ml_data[
                    "Region"
                ].unique()
            )

        )


        median_quantity = int(
            ml_data[
                "Quantity Sold"
            ].median()
        )


        new_quantity = st.number_input(

            "📦 Expected Quantity Sold",

            min_value=0,

            value=max(
                1,
                median_quantity
            )

        )


    new_data = pd.DataFrame({

        "Product": [
            new_product
        ],

        "Brand": [
            new_brand
        ],

        "Product Type": [
            new_type
        ],

        "Region": [
            new_region
        ],

        "Quantity Sold": [
            new_quantity
        ]

    })


    predicted_price = float(

        model.predict(
            new_data
        )[0]

    )


    st.markdown("")


    st.metric(

        "🔮 Predicted Listed Price",

        money(
            max(
                0,
                predicted_price
            )
        )

    )


    st.caption(

        "Prediction is based only on historical records "
        "in the uploaded dataset."

    )


    # --------------------------------------------------------
    # TEST PREDICTIONS TABLE
    # --------------------------------------------------------

    with st.expander(
        "📋 View ML Test Predictions"
    ):

        test_predictions = (
            X_test.copy()
        )

        test_predictions[
            "Actual Price"
        ] = y_test.values

        test_predictions[
            "Predicted Price"
        ] = predictions

        test_predictions[
            "Absolute Error"
        ] = np.abs(

            test_predictions[
                "Actual Price"
            ]

            -

            test_predictions[
                "Predicted Price"
            ]

        )


        st.dataframe(

            test_predictions,

            use_container_width=True,

            hide_index=True

        )


        st.download_button(

            "⬇️ Download ML Predictions",

            test_predictions
            .to_csv(
                index=False
            )
            .encode("utf-8"),

            file_name=(
                "ml_price_predictions.csv"
            ),

            mime="text/csv"

        )


# ============================================================
# PAGE 6 — DATA QUALITY
# ============================================================

elif page == "🧹 Data Quality":

    st.markdown(

        '<div class="section-title">'
        'Data Quality & Cleaning'
        '</div>',

        unsafe_allow_html=True

    )


    original_records = len(
        raw_df
    )

    valid_records = len(
        data
    )

    removed_records = (

        original_records
        -
        valid_records

    )


    duplicate_rows = int(
        raw_df.duplicated().sum()
    )


    missing_cells = int(
        raw_df.isna()
        .sum()
        .sum()
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(

        "Original Records",

        f"{original_records:,}"

    )


    c2.metric(

        "Valid Records",

        f"{valid_records:,}"

    )


    c3.metric(

        "Removed",

        f"{removed_records:,}"

    )


    c4.metric(

        "Duplicate Rows",

        f"{duplicate_rows:,}"

    )


    st.markdown(
        "#### 🔍 Column Quality"
    )


    quality = pd.DataFrame({

        "Column":

        raw_df.columns,

        "Data Type":

        raw_df.dtypes
        .astype(str)
        .values,

        "Missing Values":

        raw_df.isna()
        .sum()
        .values,

        "Unique Values":

        raw_df.nunique(
            dropna=True
        )
        .values

    })


    st.dataframe(

        quality,

        use_container_width=True,

        hide_index=True

    )


    negative_lead = int(

        (
            data[
                "Dispatch Lead Time (Days)"
            ]
            < 0
        )
        .sum()

    )


    st.markdown(
        "#### ⚠️ Data Validation"
    )


    if negative_lead > 0:

        st.warning(

            f"{negative_lead:,} records have a "
            "negative dispatch lead time. "
            "Check whether inward/dispatch dates are correct."

        )

    else:

        st.success(
            "No negative dispatch lead times detected."
        )


    if duplicate_rows > 0:

        st.warning(

            f"{duplicate_rows:,} duplicate rows "
            "were found in the original dataset."

        )

    else:

        st.success(
            "No duplicate rows detected."
        )


    st.markdown(
        "#### 📋 Cleaned Dataset"
    )


    st.dataframe(

        data.head(100),

        use_container_width=True,

        hide_index=True

    )


    st.download_button(

        "⬇️ Download Cleaned Dataset",

        data
        .to_csv(
            index=False
        )
        .encode("utf-8"),

        file_name=(
            "mobile_laptop_sales_cleaned.csv"
        ),

        mime="text/csv"

    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(

    """
    <div style="
        text-align:center;
        color:#667085;
        font-size:12px;
        padding:10px;
    ">

    <b>TechPulse Analytics</b>
    · Mobile & Laptop Sales Intelligence
    · Data Analytics + Machine Learning Project

    </div>
    """,

    unsafe_allow_html=True

)