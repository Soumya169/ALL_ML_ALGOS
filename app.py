import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ML Libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Supply Chain ML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .best-model-card {
        background: linear-gradient(135deg, #ffd700 0%, #ffa500 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_dataset(num_records=10000):
    """Generate realistic supply chain dataset"""
    np.random.seed(42)
    random.seed(42)
    
    products = [f'PROD_{i:04d}' for i in range(1, 51)]
    categories = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys']
    warehouses = ['WH_North', 'WH_South', 'WH_East', 'WH_West', 'WH_Central']
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data = []
    
    for _ in range(num_records):
        date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        product_id = random.choice(products)
        category = random.choice(categories)
        warehouse = random.choice(warehouses)
        
        base_demand = random.randint(10, 100)
        month = date.month
        seasonal_factor = 1.5 if month in [11, 12] else (1.2 if month in [6, 7, 8] else 1.0)
        day_of_week = date.weekday()
        weekday_factor = 1.0 if day_of_week < 5 else 0.7
        promotion = random.choice([0, 1]) if random.random() > 0.7 else 0
        promotion_factor = 1.3 if promotion else 1.0
        
        quantity = int(base_demand * seasonal_factor * weekday_factor * promotion_factor)
        quantity += random.randint(-5, 5)
        quantity = max(0, quantity)
        
        base_price = random.uniform(10, 500)
        discount = 0.1 if promotion else 0
        price = base_price * (1 - discount)
        
        data.append({
            'date': date,
            'product_id': product_id,
            'category': category,
            'warehouse': warehouse,
            'quantity_sold': quantity,
            'price': round(price, 2),
            'promotion': promotion,
            'stock_level': random.randint(0, 500),
            'lead_time': random.randint(1, 14),
            'supplier_reliability': round(random.uniform(0.7, 1.0), 2),
            'day_of_week': day_of_week,
            'month': month,
            'year': date.year,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_holiday_season': 1 if month in [11, 12] else 0
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    return df

########### FEATURE ENGINEERING ################
@st.cache_data
def create_features(df):
    """Feature engineering"""
    df = df.copy()
    df = df.sort_values(['product_id', 'date'])
    
    # Lag features
    df['quantity_lag_1'] = df.groupby('product_id')['quantity_sold'].shift(1)
    df['quantity_lag_7'] = df.groupby('product_id')['quantity_sold'].shift(7)
    
    # Rolling features
    df['quantity_rolling_mean_7'] = df.groupby('product_id')['quantity_sold'].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    
    df['quantity_rolling_std_7'] = df.groupby('product_id')['quantity_sold'].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    
    # Price features
    df['price_to_avg_ratio'] = df['price'] / df.groupby('category')['price'].transform('mean')
    
    # Interaction features
    df['promo_weekend'] = df['promotion'] * df['is_weekend']
    df['stock_to_demand_ratio'] = df['stock_level'] / (df['quantity_lag_1'] + 1)

    df = df.fillna(0)
    return df

##################### MODEL TRAINING #################
@st.cache_resource(show_spinner=False)
def train_all_models(X_train, X_test, y_train, y_test):
    """Train all ML models and return results"""
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100
        
        results.append({
            'model': name,
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'test_mae': round(test_mae, 2),
            'test_r2': round(test_r2, 4),
            'test_mape': round(test_mape, 2)
        })
        
        trained_models[name] = {
            'model': model,
            'predictions': y_pred_test,
            'metrics': results[-1]
        }
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("✅ All models trained successfully!")
    progress_bar.empty()
    
    return pd.DataFrame(results).sort_values('test_rmse'), trained_models

####################### MAIN APP ##################
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Smart Supply Chain Forecasting Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Demand Prediction with 9 ML Algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    dataset_size = st.sidebar.slider("Dataset Size", 1000, 50000, 10000, 1000)

    st.sidebar.subheader("🔧 Features")
    use_advanced_features = st.sidebar.checkbox("Use Advanced Features", value=True)
    
    page = st.sidebar.radio("📑 Navigation", 
                            ["ℹ️ About","📊 Predict Demand", "🤖 Model Comparison", "📈 Data Summary"])
    
    # Generate and prepare data
    with st.spinner("🔄 Generating dataset..."):
        df = generate_dataset(dataset_size)
    
    with st.spinner("🔄 Engineering features..."):
        df = create_features(df)
    
    # Prepare data for modeling
    feature_cols = [
        'price', 'promotion', 'stock_level', 'lead_time', 
        'supplier_reliability', 'day_of_week', 'month',
        'is_weekend', 'is_holiday_season'
    ]
    
    if use_advanced_features:
        feature_cols.extend([
            'quantity_lag_1', 'quantity_lag_7',
            'quantity_rolling_mean_7', 'quantity_rolling_std_7',
            'price_to_avg_ratio', 'promo_weekend', 'stock_to_demand_ratio'
        ])
    
    # Encode categoricals
    le_category = LabelEncoder()
    le_warehouse = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['warehouse_encoded'] = le_warehouse.fit_transform(df['warehouse'])
    
    feature_cols.extend(['category_encoded', 'warehouse_encoded'])
    
    # Prepare data
    df_clean = df.dropna(subset=feature_cols + ['quantity_sold'])
    X = df_clean[feature_cols]
    y = df_clean['quantity_sold']
    
    # Split and scale
    df_clean = df_clean.sort_values('date')

    split_date = df_clean['date'].quantile(0.8)

    train_idx = df_clean['date'] <= split_date
    test_idx  = df_clean['date'] > split_date

    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models (cached)
    results_df, trained_models = train_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    if page == "ℹ️ About":
        show_about_page()
        
    elif page == "📊 Predict Demand":
        show_prediction_page(df, trained_models, results_df, scaler, le_category, le_warehouse, feature_cols)
    
    elif page == "🤖 Model Comparison":
        show_model_comparison_page(results_df, trained_models, y_test)
    
    elif page == "📈 Data Summary":
        show_data_summary_page(df)
    
    

def show_prediction_page(df, trained_models, results_df, scaler, le_category, le_warehouse, feature_cols):
    st.header("📊 Demand Forecasting")
    st.write("Enter product and market conditions to predict demand")
    
    # Select best model
    best_model_name = results_df.iloc[0]['model']
    best_model = trained_models[best_model_name]['model']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Input Features")
        
        # Create form
        with st.form("prediction_form"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("**📦 Product Info**")
                category = st.selectbox("Category", ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys'])
                price = st.number_input("Price ($)", 10.0, 500.0, 150.0, 10.0)
                promotion = st.selectbox("Promotion", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
            with col_b:
                st.markdown("**🏭 Warehouse**")
                warehouse = st.selectbox("Warehouse", ['WH_North', 'WH_South', 'WH_East', 'WH_West', 'WH_Central'])
                stock_level = st.number_input("Stock Level", 0, 1000, 200)
                lead_time = st.number_input("Lead Time (days)", 1, 30, 7)
            
            with col_c:
                st.markdown("**📅 Time Features**")
                month = st.slider("Month", 1, 12, 11)
                day_of_week = st.slider("Day of Week", 0, 6, 3)
                is_weekend = 1 if day_of_week >= 5 else 0
                is_holiday = st.selectbox("Holiday Season", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
            supplier_reliability = st.slider("Supplier Reliability", 0.0, 1.0, 0.9, 0.05)
            
            submit = st.form_submit_button("🚀 Predict Demand", use_container_width=True)
        
        if submit:
            # Create feature vector
            input_data = {
                'price': price,
                'promotion': promotion,
                'stock_level': stock_level,
                'lead_time': lead_time,
                'supplier_reliability': supplier_reliability,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': is_weekend,
                'is_holiday_season': is_holiday,
                'quantity_lag_1': df['quantity_sold'].mean(),
                'quantity_lag_7': df['quantity_sold'].mean(),
                'quantity_rolling_mean_7': df['quantity_sold'].mean(),
                'quantity_rolling_std_7': df['quantity_sold'].std(),

                'price_to_avg_ratio': 1.0,
                'promo_weekend': promotion * is_weekend,
                'stock_to_demand_ratio': stock_level / 50,
                'category_encoded': le_category.transform([category])[0],
                'warehouse_encoded': le_warehouse.transform([warehouse])[0]
            }
            
            # Create feature array
            X_input = np.array([[input_data.get(col, 0) for col in feature_cols]])
            X_input_scaled = scaler.transform(X_input)
            
            # Predict
            prediction = best_model.predict(X_input_scaled)[0]
            
            # Display result
            st.success("✅ Prediction Complete!")
            
            # Prediction card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
                <h3>Predicted Demand</h3>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:.0f}</h1>
                <p style="font-size: 1.2rem;">units</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction > stock_level:
                st.warning(f"⚠️ Predicted demand ({prediction:.0f}) exceeds current stock ({stock_level}). Consider reordering.")
            else:
                st.info(f"ℹ️ Stock level is sufficient for predicted demand.")
            
            if promotion == 1:
                st.success("✓ Promotion active - expect higher demand.")
            
            # Show model used
            st.info(f"🤖 Prediction made using: **{best_model_name}** (R² = {results_df.iloc[0]['test_r2']:.4f})")
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        avg_quantity = df['quantity_sold'].mean()
        avg_price = df['price'].mean()
        promo_rate = df['promotion'].mean() * 100
        
        st.metric("Avg Quantity Sold", f"{avg_quantity:.0f}", "units")
        st.metric("Avg Price", f"${avg_price:.2f}")
        st.metric("Promotion Rate", f"{promo_rate:.1f}%")
        
        # Historical trend
        st.subheader("📈 Historical Trend")
        daily_sales = df.groupby('date')['quantity_sold'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='quantity_sold', 
                     title='Daily Sales Volume')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ==================== MODEL COMPARISON PAGE ====================
def show_model_comparison_page(results_df, trained_models, y_test):
    st.header("🤖 Model Performance Comparison")
    
    # Best model highlight
    best_model = results_df.iloc[0]
    
    st.markdown(f"""
    <div class="best-model-card">
        <h2>🏆 Best Performing Model</h2>
        <h1 style="font-size: 2.5rem; margin: 1rem 0; color: #333;">{best_model['model']}</h1>
        <div style="display: flex; justify-content: space-around; margin-top: 1.5rem;">
            <div>
                <p style="color: #666; margin-bottom: 0.5rem;">Test RMSE</p>
                <h3 style="color: #333;">{best_model['test_rmse']}</h3>
            </div>
            <div>
                <p style="color: #666; margin-bottom: 0.5rem;">R² Score</p>
                <h3 style="color: #333;">{best_model['test_r2']}</h3>
            </div>
            <div>
                <p style="color: #666; margin-bottom: 0.5rem;">MAE</p>
                <h3 style="color: #333;">{best_model['test_mae']}</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    st.subheader("📊 All Models Comparison")
    
    # Format dataframe for display
    display_df = results_df.copy()
    display_df = display_df.style.background_gradient(subset=['test_rmse'], cmap='RdYlGn_r')\
                                  .background_gradient(subset=['test_r2'], cmap='RdYlGn')\
                                  .format({'test_rmse': '{:.2f}', 'test_mae': '{:.2f}', 
                                          'test_r2': '{:.4f}', 'test_mape': '{:.2f}%'})
    
    st.dataframe(display_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # RMSE comparison
        fig_rmse = px.bar(results_df, x='model', y='test_rmse', 
                         title='Test RMSE Comparison (Lower is Better)',
                         color='test_rmse', color_continuous_scale='RdYlGn_r')
        fig_rmse.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        # R² comparison
        fig_r2 = px.bar(results_df.sort_values('test_r2', ascending=False), 
                       x='model', y='test_r2',
                       title='R² Score Comparison (Higher is Better)',
                       color='test_r2', color_continuous_scale='RdYlGn')
        fig_r2.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Prediction vs Actual scatter plots
    st.subheader("📈 Prediction Quality Visualization")
    
    selected_models = st.multiselect(
        "Select models to compare",
        list(trained_models.keys()),
        default=[results_df.iloc[0]['model'], results_df.iloc[1]['model']]
    )
    
    if selected_models:
        fig = make_subplots(rows=1, cols=len(selected_models),
                           subplot_titles=selected_models)
        
        for idx, model_name in enumerate(selected_models, 1):
            y_pred = trained_models[model_name]['predictions']
            
            # Sample for cleaner visualization
            sample_indices = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
            y_test_sample = y_test.iloc[sample_indices]
            y_pred_sample = y_pred[sample_indices]
            
            fig.add_trace(
                go.Scatter(x=y_test_sample, y=y_pred_sample, 
                          mode='markers', name=model_name,
                          marker=dict(size=5, opacity=0.6)),
                row=1, col=idx
            )
            
            # Perfect prediction line
            min_val = min(y_test_sample.min(), y_pred_sample.min())
            max_val = max(y_test_sample.max(), y_pred_sample.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', name='Perfect Prediction',
                          line=dict(color='red', dash='dash')),
                row=1, col=idx
            )
            
            fig.update_xaxes(title_text="Actual", row=1, col=idx)
            fig.update_yaxes(title_text="Predicted", row=1, col=idx)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Best Accuracy**
        
        {best_model['model']} achieves the lowest RMSE of {best_model['test_rmse']:.2f}, 
        making it the most accurate model.
        """)
    
    with col2:
        st.info(f"""
        **Model Count**
        
        {len(results_df)} different algorithms were trained and evaluated, 
        including tree-based, linear, and ensemble methods.
        """)
    
    with col3:
        avg_rmse = results_df['test_rmse'].mean()
        st.info(f"""
        **Average Performance**
        
        Average RMSE across all models: {avg_rmse:.2f}
        Standard deviation: {results_df['test_rmse'].std():.2f}
        """)

# ==================== DATA SUMMARY PAGE ====================
def show_data_summary_page(df):
    st.header("📈 Dataset Overview")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df['quantity_sold'].sum():,.0f}</h3>
            <p>Total Units Sold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${df['price'].mean():.2f}</h3>
            <p>Average Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{df['promotion'].mean()*100:.1f}%</h3>
            <p>Promotion Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Date range
    st.subheader("📅 Data Timeline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Date", df['date'].min().strftime('%Y-%m-%d'))
    col2.metric("End Date", df['date'].max().strftime('%Y-%m-%d'))
    col3.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
    
    # Distribution visualizations
    st.subheader("📊 Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = df['category'].value_counts()
        fig_cat = px.pie(values=category_counts.values, names=category_counts.index,
                        title='Product Category Distribution')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Warehouse distribution
        warehouse_counts = df['warehouse'].value_counts()
        fig_wh = px.bar(x=warehouse_counts.index, y=warehouse_counts.values,
                       title='Warehouse Distribution',
                       labels={'x': 'Warehouse', 'y': 'Count'})
        st.plotly_chart(fig_wh, use_container_width=True)
    
    # Time series analysis
    st.subheader("📈 Time Series Analysis")
    
    daily_stats = df.groupby('date').agg({
        'quantity_sold': 'sum',
        'price': 'mean',
        'promotion': 'mean'
    }).reset_index()
    
    fig_ts = make_subplots(rows=2, cols=1, 
                          subplot_titles=('Daily Sales Volume', 'Average Price & Promotion Rate'))
    
    fig_ts.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['quantity_sold'], 
                  name='Quantity Sold', line=dict(color='#667eea')),
        row=1, col=1
    )
    
    fig_ts.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['price'], 
                  name='Avg Price', line=dict(color='#10b981')),
        row=2, col=1
    )
    
    fig_ts.update_xaxes(title_text="Date", row=2, col=1)
    fig_ts.update_yaxes(title_text="Quantity", row=1, col=1)
    fig_ts.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig_ts.update_layout(height=600, showlegend=True)
    
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(df[['quantity_sold', 'price', 'stock_level', 'lead_time', 
                     'supplier_reliability']].describe(), use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlations")
    numeric_cols = ['quantity_sold', 'price', 'promotion', 'stock_level', 
                    'lead_time', 'supplier_reliability', 'is_weekend', 'is_holiday_season']
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        title='Feature Correlation Matrix')
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(100), use_container_width=True)

# ==================== ABOUT PAGE ====================
def show_about_page():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🚀 Smart Supply Chain Forecasting Platform
    
    A comprehensive machine learning application for demand forecasting in supply chain management.
    
    ### 🎯 Features
    
    - **9 ML Algorithms**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, 
      Gradient Boosting, XGBoost, SVR, and KNN
    - **Real-time Predictions**: Interactive form for demand forecasting
    - **Model Comparison**: Side-by-side performance metrics and visualizations
    - **Data Analysis**: Comprehensive dataset exploration and insights
    - **Feature Engineering**: Advanced features including lag variables and rolling statistics
    
    ### 📊 Dataset Features
    
    - **Time-series data** with seasonal patterns
    - **Promotion and pricing** effects
    - **Inventory and stock** levels
    - **Supplier reliability** metrics
    - **Holiday and weekend** indicators
    - **Multi-warehouse** logistics data
    
    ### 🤖 Machine Learning Models
    
    **Linear Models:**
    - Linear Regression (baseline)
    - Ridge Regression (L2 regularization)
    - Lasso Regression (L1 regularization)
    
    **Tree-Based Models:**
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost
    
    **Other Algorithms:**
    - Support Vector Regression (SVR)
    - K-Nearest Neighbors (KNN)
    
    ### 📈 Evaluation Metrics
    
    - **RMSE** (Root Mean Squared Error): Measures prediction accuracy
    - **MAE** (Mean Absolute Error): Average prediction error
    - **R²** (R-squared): Goodness of fit
    - **MAPE** (Mean Absolute Percentage Error): Percentage error
    
    ### 🛠️ Technology Stack
    
    - **Framework**: Streamlit
    - **ML Libraries**: scikit-learn, XGBoost
    - **Data Processing**: pandas, NumPy
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Language**: Python 3.8+
    
    ### 💼 Use Cases
    
    - Demand forecasting for retail
    - Inventory optimization
    - Stock level prediction
    - Supply chain planning
    - Trend analysis
    
    ### 📝 How to Use
    
    1. **Predict Demand**: Enter product details and get instant demand predictions
    2. **Compare Models**: Analyze performance across all ML algorithms
    3. **Explore Data**: Understand dataset characteristics and patterns
    
    ### 🎓 Resume Highlights
    
    This project demonstrates:
    - ✅ Machine Learning model development and deployment
    - ✅ Data preprocessing and feature engineering
    - ✅ Interactive web application development
    - ✅ Data visualization and storytelling
    - ✅ Production-ready code architecture
    
    ### 📧 Contact & Links
    
    - **GitHub**: [Your GitHub Profile]
    - **LinkedIn**: [Your LinkedIn]
    - **Portfolio**: [Your Portfolio Site]
    
    ---
    
    **Built with ❤️ using Streamlit**
    
    Version 1.0.0 | Last Updated: December 2024
    """)
    
    # Technical details in expander
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        ### Data Generation
        - Synthetic dataset with realistic patterns
        - Seasonal trends (holiday season boost)
        - Day-of-week effects
        - Promotion impacts
        - Random noise for realism
        
        ### Feature Engineering
        - **Lag Features**: Previous day, week values
        - **Rolling Statistics**: 7-day moving averages
        - **Ratio Features**: Price-to-average comparisons
        - **Interaction Terms**: Promotion × Weekend
        
        ### Model Training
        - Train/test split (80/20)
        - Feature scaling with StandardScaler
        - Label encoding for categorical variables
        - Cross-validation ready
        
        ### Performance Optimization
        - Streamlit caching for data and models
        - Efficient numpy operations
        - Lazy loading of visualizations
        """)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### Installation
        
        ```bash
        # Install dependencies
        pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn plotly
        
        # Run the app
        streamlit run app.py
        ```
        
        ### Configuration
        
        Use the sidebar to:
        - Adjust dataset size (1K - 50K records)
        - Enable/disable advanced features
        - Navigate between pages
        
        ### Making Predictions
        
        1. Go to "Predict Demand" page
        2. Fill in the form with product details
        3. Click "Predict Demand"
        4. Review recommendations
        
        ### Comparing Models
        
        1. Navigate to "Model Comparison"
        2. View best performing model
        3. Analyze comparison table
        4. Select models for visualization
        
        ### Exploring Data
        
        1. Visit "Data Summary" page
        2. Review key statistics
        3. Analyze distributions
        4. Examine time series trends
        """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()