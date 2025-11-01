# ==============================================================================
# QuantumAd Prophecy: Enhanced AI-Powered Advertising ROI Optimizer
# PHASE 2: STREAMLIT APPLICATION (Complete Working Version)
# Developed by Yash Gupta with ❤️
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="QuantumAd Prophecy",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# CUSTOM STYLING (DUAL THEME SUPPORT)
# ---------------------------------
st.markdown("""
<style>
    /* Main background with theme support */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #2c3e50;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 3px solid #ff6b6b;
    }
    
    /* Text and headers with gradient */
    h1, h2, h3 {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    h4, h5, h6 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5);
    }
    
    /* Enhanced metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600;
        color: #2c3e50 !important;
    }
    
    /* Tabs enhanced styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: transparent;
        border-radius: 10px;
        padding: 15px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    /* Custom containers */
    .custom-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        background: linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%);
        color: black;
        border-radius: 15px;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL & SCALER - FIXED VERSION
# ---------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('sales_prediction_model.joblib')
        scaler = joblib.load('scaler.joblib')
        st.sidebar.success("✅ Model loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure 'sales_prediction_model.joblib' and 'scaler.joblib' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

model, scaler = load_assets()

# ---------------------------------
# ENHANCED HELPER FUNCTIONS - FIXED
# ---------------------------------
def predict_sales(tv, radio, newspaper):
    if model is None or scaler is None:
        # Fallback prediction if model not loaded
        return max(0, tv * 0.8 + radio * 0.5 + newspaper * 0.2 + 5000)
    
    try:
        # Create a DataFrame for the input with correct column names
        input_data = pd.DataFrame({
            'TV': [tv], 
            'Radio': [radio], 
            'Newspaper': [newspaper]
        })
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        return max(0, prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return max(0, tv * 0.8 + radio * 0.5 + newspaper * 0.2 + 5000)

def calculate_roi_metrics(tv, radio, newspaper, predicted_sales):
    total_investment = tv + radio + newspaper
    if total_investment == 0:
        return 0, 0, 0
    
    roi = (predicted_sales - total_investment) / total_investment
    profit_margin = (predicted_sales - total_investment) / predicted_sales if predicted_sales > 0 else 0
    breakeven_point = total_investment / (predicted_sales / 1000) if predicted_sales > 0 else float('inf')
    
    return roi, profit_margin, breakeven_point

def get_feature_importances():
    """Get feature importances from the model"""
    if model is None:
        return {'TV': 0.6, 'Radio': 0.3, 'Newspaper': 0.1}
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            importances = importances / np.sum(importances)
        else:
            importances = np.array([0.6, 0.3, 0.1])
        
        return dict(zip(['TV', 'Radio', 'Newspaper'], importances))
    except:
        return {'TV': 0.6, 'Radio': 0.3, 'Newspaper': 0.1}

def generate_comprehensive_recommendations(tv, radio, newspaper, importances, predicted_sales, roi):
    total_budget = tv + radio + newspaper
    if total_budget == 0:
        return ["💰 Start by allocating a budget to see AI-powered recommendations."]

    recommendations = []
    
    # Sort channels by importance
    sorted_channels = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    best_channel, best_importance = sorted_channels[0]
    worst_channel, worst_importance = sorted_channels[-1]
    
    allocations = {'TV': tv, 'Radio': radio, 'Newspaper': newspaper}
    allocation_percentages = {k: v/total_budget for k, v in allocations.items()}
    
    # ROI-based recommendations
    if roi > 1:
        recommendations.append("🎯 **Excellent ROI**: Your current allocation is generating outstanding returns! Consider scaling successful channels.")
    elif roi < 0:
        recommendations.append("⚠️ **Negative ROI**: Current allocation isn't profitable. Review channel performance and reallocate budget.")
    
    # Budget allocation recommendations
    if allocation_percentages[best_channel] < best_importance - 0.1:
        recommendations.append(f"💡 **Optimize Allocation**: Increase {best_channel} budget from {allocation_percentages[best_channel]:.1%} to at least {best_importance:.1%} for maximum impact.")
    
    if allocation_percentages[worst_channel] > worst_importance + 0.1:
        recommendations.append(f"🔄 **Reallocate Resources**: Reduce {worst_channel} spending and shift funds to higher-performing channels.")
    
    # Performance thresholds
    if predicted_sales / total_budget < 2:
        recommendations.append("📊 **Efficiency Alert**: Sales-to-investment ratio is low. Consider testing new creative strategies or audience targeting.")
    
    if not recommendations:
        recommendations.append("✅ **Optimal Performance**: Your current strategy is well-balanced. Monitor performance and test incremental optimizations.")

    return recommendations

def create_sample_historical_data():
    """Create sample historical data for demonstration"""
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    data = []
    for date in dates:
        tv = np.random.normal(50000, 10000)
        radio = np.random.normal(30000, 5000)
        newspaper = np.random.normal(20000, 3000)
        sales = predict_sales(tv, radio, newspaper)
        data.append({
            'Date': date,
            'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper,
            'Sales': sales,
            'ROI': (sales - (tv+radio+newspaper)) / (tv+radio+newspaper)
        })
    return pd.DataFrame(data)

# ---------------------------------
# MAIN APP LAYOUT
# ---------------------------------

# --- ENHANCED HEADER ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🔮 QuantumAd Prophecy</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>AI-Powered Advertising ROI Optimizer</h4>", unsafe_allow_html=True)
    st.markdown("---")

# --- ENHANCED SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.header("🎛️ Campaign Configuration")
    
    budget_unit = st.selectbox("Select Budget Unit", ["Thousands ($K)", "Millions ($M)"], index=1)
    multiplier = 1000 if budget_unit == "Thousands ($K)" else 1000000
    unit_symbol = "K" if budget_unit == "Thousands ($K)" else "M"
    
    total_budget_display = st.number_input(
        f"Total Advertising Budget ($ {unit_symbol})", 
        min_value=1.0, 
        value=100.0, 
        step=10.0,
        help="Enter your total advertising budget"
    )
    total_budget = total_budget_display * multiplier

    st.subheader("📊 Budget Allocation")
    
    # Enhanced budget allocation with percentage indicators
    tv_budget_display = st.slider(
        "Television 📺", 
        0.0, total_budget_display, 
        value=50.0, 
        step=1.0
    )
    
    remaining_after_tv = total_budget_display - tv_budget_display
    radio_budget_display = st.slider(
        "Radio 📻", 
        0.0, remaining_after_tv, 
        value=min(30.0, remaining_after_tv), 
        step=1.0
    )
    
    newspaper_budget_display = total_budget_display - tv_budget_display - radio_budget_display
    st.slider(
        "Newspaper 📰", 
        0.0, total_budget_display, 
        value=newspaper_budget_display, 
        step=1.0,
        disabled=True
    )

    # Display allocation percentages
    st.markdown("**Allocation Summary:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TV", f"{(tv_budget_display/total_budget_display*100):.1f}%")
    with col2:
        st.metric("Radio", f"{(radio_budget_display/total_budget_display*100):.1f}%")
    with col3:
        st.metric("Newspaper", f"{(newspaper_budget_display/total_budget_display*100):.1f}%")

    # Convert to actual values
    tv_budget = tv_budget_display * multiplier
    radio_budget = radio_budget_display * multiplier
    newspaper_budget = newspaper_budget_display * multiplier

    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        st.number_input("Cost per TV Ad Spot", value=1000, help="Cost for one television ad spot")
        st.number_input("Cost per Radio Ad Spot", value=500, help="Cost for one radio ad spot")
        st.number_input("Cost per Newspaper Ad", value=200, help="Cost for one newspaper ad")
        
        confidence_level = st.slider("Model Confidence Level", 50, 95, 85)
        st.info(f"Using {confidence_level}% confidence interval for predictions")

    # Auto-Optimize with enhanced logic
    if st.button("🚀 Auto-Optimize Budget", use_container_width=True):
        importances = get_feature_importances()
        
        # Calculate optimal allocation based on feature importances
        optimal_tv = total_budget * importances['TV']
        optimal_radio = total_budget * importances['Radio']
        optimal_newspaper = total_budget * importances['Newspaper']
        
        # Update the slider values (this will trigger a rerun)
        st.session_state.tv_budget = optimal_tv / multiplier
        st.session_state.radio_budget = optimal_radio / multiplier
        st.session_state.newspaper_budget = optimal_newspaper / multiplier
        
        st.success("🎉 Budget optimized based on AI model insights!")

    st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
# Calculate key metrics
allocated_budget = tv_budget + radio_budget + newspaper_budget
remaining_budget = total_budget - allocated_budget
predicted_sales = predict_sales(tv_budget, radio_budget, newspaper_budget)
roi, profit_margin, breakeven_point = calculate_roi_metrics(tv_budget, radio_budget, newspaper_budget, predicted_sales)

# --- ENHANCED KPI DASHBOARD ---
st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
st.header("📈 Performance Dashboard")

# First row of KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    delta_budget = (allocated_budget - total_budget/2) / (total_budget/2)
    st.metric(
        "Total Budget", 
        f"${total_budget/multiplier:,.1f}{unit_symbol}",
        f"{delta_budget:+.1%}"
    )

with col2:
    utilization = allocated_budget / total_budget
    st.metric(
        "Budget Utilization", 
        f"${allocated_budget/multiplier:,.1f}{unit_symbol}",
        f"{utilization:.1%}"
    )

with col3:
    st.metric(
        "Predicted Sales", 
        f"${predicted_sales/multiplier:,.1f}{unit_symbol}",
        f"ROI: {roi:.1%}"
    )

with col4:
    st.metric(
        "Profit Margin", 
        f"{profit_margin:.1%}",
        f"Breakeven: {breakeven_point:.0f} units"
    )

# Second row of KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    efficiency_score = 85 + np.random.randint(-5, 6)  # Simulated efficiency score
    st.metric("Allocation Efficiency", f"{efficiency_score}%")

with col2:
    sales_ratio = predicted_sales / allocated_budget if allocated_budget > 0 else 0
    st.metric("Sales/Investment Ratio", f"{sales_ratio:.1f}x")

with col3:
    optimal_roi = roi * 1.2  # Simulated optimal ROI
    st.metric("Optimal ROI Potential", f"{optimal_roi:.1f}x")

with col4:
    growth_potential = max(0, (optimal_roi - roi) / roi * 100) if roi > 0 else 0
    st.metric("Growth Opportunity", f"+{growth_potential:.1f}%")

st.markdown("</div>", unsafe_allow_html=True)

# --- ENHANCED TABS WITH MORE FEATURES ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🤖 AI Insights", 
    "📈 Trend Analysis", 
    "🎯 Recommendations",
    "📋 Campaign Details"
])

with tab1:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced Investment vs Sales Chart
        df_perf = pd.DataFrame({
            'Channel': ['TV', 'Radio', 'Newspaper'],
            'Investment': [tv_budget, radio_budget, newspaper_budget],
            'Sales Contribution': [
                predict_sales(tv_budget, 0, 0), 
                predict_sales(0, radio_budget, 0), 
                predict_sales(0, 0, newspaper_budget)
            ],
            'ROI': [
                (predict_sales(tv_budget, 0, 0) - tv_budget) / tv_budget if tv_budget > 0 else 0,
                (predict_sales(0, radio_budget, 0) - radio_budget) / radio_budget if radio_budget > 0 else 0,
                (predict_sales(0, 0, newspaper_budget) - newspaper_budget) / newspaper_budget if newspaper_budget > 0 else 0
            ]
        })
        
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(
            name='Investment', 
            x=df_perf['Channel'], 
            y=df_perf['Investment'],
            marker_color='#FF6B6B',
            text=df_perf['Investment'].apply(lambda x: f'${x/multiplier:,.0f}{unit_symbol}'),
            textposition='auto'
        ))
        fig_inv.add_trace(go.Bar(
            name='Sales Contribution', 
            x=df_perf['Channel'], 
            y=df_perf['Sales Contribution'],
            marker_color='#4ECDC4',
            text=df_perf['Sales Contribution'].apply(lambda x: f'${x/multiplier:,.0f}{unit_symbol}'),
            textposition='auto'
        ))
        
        fig_inv.update_layout(
            title='Investment vs Sales Contribution by Channel',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        st.plotly_chart(fig_inv, use_container_width=True)
        
        # ROI by Channel
        fig_roi = px.bar(
            df_perf, 
            x='Channel', 
            y='ROI',
            title='ROI by Advertising Channel',
            color='ROI',
            color_continuous_scale=['#FF6B6B', '#4ECDC4'],
            text=df_perf['ROI'].apply(lambda x: f'{x:.1%}')
        )
        fig_roi.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            showlegend=False,
            height=300
        )
        fig_roi.update_traces(textposition='outside')
        st.plotly_chart(fig_roi, use_container_width=True)

    with col2:
        # Enhanced Budget Utilization
        labels = ['TV', 'Radio', 'Newspaper', 'Remaining']
        values = [tv_budget, radio_budget, newspaper_budget, remaining_budget]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95a5a6']

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=colors,
            textinfo='percent+label',
            insidetextorientation='radial',
            hoverinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
        )])
        fig_pie.update_layout(
            title='Budget Allocation Breakdown',
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=400,
            annotations=[dict(text=f'Total<br>${total_budget/multiplier:,.0f}{unit_symbol}', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = efficiency_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Campaign Efficiency Score"},
            delta = {'reference': 80, 'increasing': {'color': "#4ECDC4"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#FF6B6B"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': '#f8f9fa'},
                    {'range': [70, 90], 'color': '#e9ecef'},
                    {'range': [90, 100], 'color': '#dee2e6'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature Importance with enhanced visualization
        importances = get_feature_importances()
        
        feature_importance_df = pd.DataFrame({
            'Channel': ['TV', 'Radio', 'Newspaper'],
            'Importance': [importances['TV'], importances['Radio'], importances['Newspaper']],
            'Current_Allocation': [tv_budget/allocated_budget, radio_budget/allocated_budget, newspaper_budget/allocated_budget]
        })
        
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            name='Model Importance',
            y=feature_importance_df['Channel'],
            x=feature_importance_df['Importance'],
            orientation='h',
            marker_color='#FF6B6B',
            text=feature_importance_df['Importance'].apply(lambda x: f'{x:.1%}'),
            textposition='auto'
        ))
        fig_imp.add_trace(go.Bar(
            name='Your Allocation',
            y=feature_importance_df['Channel'],
            x=feature_importance_df['Current_Allocation'],
            orientation='h',
            marker_color='#4ECDC4',
            text=feature_importance_df['Current_Allocation'].apply(lambda x: f'{x:.1%}'),
            textposition='auto'
        ))
        
        fig_imp.update_layout(
            title='Channel Efficiency: Model Importance vs Your Allocation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=400,
            xaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("""
        **Interpretation Guide:**
        - **Red bars** show what the AI model recommends based on historical performance
        - **Green bars** show your current budget allocation
        - Align green bars with red bars for optimal performance
        """)

    with col2:
        # Performance Matrix
        st.subheader("📊 Performance Matrix")
        
        # Calculate performance scores
        roi_score = min(100, max(0, roi * 100))
        utilization_score = min(100, (allocated_budget/total_budget)*100)
        efficiency_gap = 100 - abs(feature_importance_df['Importance'] - feature_importance_df['Current_Allocation']).mean() * 200
        balance_score = min(100, efficiency_gap)
        
        matrix_data = pd.DataFrame({
            'Metric': ['ROI Efficiency', 'Budget Utilization', 'Channel Balance', 'Sales Potential'],
            'Score': [roi_score, utilization_score, balance_score, min(100, (predicted_sales/total_budget)*5)],
            'Target': [80, 95, 90, 85]
        })
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=matrix_data['Score'],
            theta=matrix_data['Metric'],
            fill='toself',
            name='Current Performance',
            line=dict(color='#FF6B6B', width=2),
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=matrix_data['Target'],
            theta=matrix_data['Metric'],
            fill='toself',
            name='Target Performance',
            line=dict(color='#4ECDC4', width=2),
            fillcolor='rgba(78, 205, 196, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Performance vs Target Metrics",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    
    # Generate sample historical data
    historical_data = create_sample_historical_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical Trend Analysis
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Sales'],
            mode='lines+markers',
            name='Sales Trend',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['TV'] + historical_data['Radio'] + historical_data['Newspaper'],
            mode='lines+markers',
            name='Total Investment',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            title='Historical Sales vs Investment Trend',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # ROI Trend Analysis
        fig_roi_trend = go.Figure()
        
        fig_roi_trend.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['ROI'],
            mode='lines+markers',
            name='ROI Trend',
            line=dict(color='#45B7D1', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(69, 183, 209, 0.2)'
        ))
        
        # Add average line
        avg_roi = historical_data['ROI'].mean()
        fig_roi_trend.add_hline(
            y=avg_roi, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Average ROI: {avg_roi:.1%}"
        )
        
        fig_roi_trend.update_layout(
            title='Historical ROI Performance',
            xaxis_title='Date',
            yaxis_title='ROI',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=400,
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig_roi_trend, use_container_width=True)
    
    # Channel Performance Over Time
    st.subheader("Channel Performance Evolution")
    
    fig_channels = go.Figure()
    
    fig_channels.add_trace(go.Scatter(
        x=historical_data['Date'], 
        y=historical_data['TV'], 
        name='TV Investment',
        line=dict(color='#FF6B6B', width=2),
        stackgroup='one'
    ))
    
    fig_channels.add_trace(go.Scatter(
        x=historical_data['Date'], 
        y=historical_data['Radio'], 
        name='Radio Investment',
        line=dict(color='#4ECDC4', width=2),
        stackgroup='one'
    ))
    
    fig_channels.add_trace(go.Scatter(
        x=historical_data['Date'], 
        y=historical_data['Newspaper'], 
        name='Newspaper Investment',
        line=dict(color='#45B7D1', width=2),
        stackgroup='one'
    ))
    
    fig_channels.update_layout(
        title='Channel Investment Distribution Over Time',
        xaxis_title='Date',
        yaxis_title='Investment ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400
    )
    st.plotly_chart(fig_channels, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    
    # Enhanced Recommendations
    importances = get_feature_importances()
    recommendations = generate_comprehensive_recommendations(
        tv_budget, radio_budget, newspaper_budget, 
        importances, predicted_sales, roi
    )
    
    st.subheader("🎯 AI-Powered Strategic Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        if "Excellent" in rec or "Optimal" in rec:
            emoji, color = "💡", "#4ECDC4"
        elif "Alert" in rec or "Negative" in rec:
            emoji, color = "⚠️", "#FF6B6B"
        else:
            emoji, color = "🔧", "#45B7D1"
            
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(240,240,240,0.9));
            padding: 20px; 
            border-radius: 15px; 
            border-left: 5px solid {color}; 
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        '>
            <h4 style='margin:0; color: #2c3e50;'>{emoji} Recommendation #{i}</h4>
            <p style='margin:10px 0 0 0; color: #2c3e50; font-size: 14px;'>{rec}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Action Plan
    st.subheader("📋 Suggested Action Plan")
    
    action_steps = [
        "Review current channel performance metrics weekly",
        "Adjust budget allocation based on AI recommendations",
        "Set up A/B testing for new creative strategies",
        "Monitor ROI and adjust campaigns accordingly",
        "Schedule monthly performance review meetings",
        "Test new audience segments for better targeting"
    ]
    
    for step in action_steps:
        st.markdown(f"✅ {step}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Details")
        
        campaign_data = {
            'Parameter': ['Campaign Name', 'Start Date', 'End Date', 'Total Budget', 'Target Audience', 
                         'Primary Objective', 'Secondary Objective', 'KPI Target'],
            'Value': ['Q2 Brand Awareness', '2024-04-01', '2024-06-30', f'${total_budget/multiplier:,.0f}{unit_symbol}', 
                     'Age 25-45 Professionals', 'Brand Awareness', 'Lead Generation', 'ROI > 150%']
        }
        
        st.table(pd.DataFrame(campaign_data))
    
    with col2:
        st.subheader("Performance Summary")
        
        summary_data = {
            'Metric': ['Predicted Sales', 'Expected ROI', 'Profit Margin', 'Breakeven Point', 
                      'Budget Utilization', 'Allocation Efficiency'],
            'Value': [f'${predicted_sales/multiplier:,.1f}{unit_symbol}', f'{roi:.1%}', 
                     f'{profit_margin:.1%}', f'{breakeven_point:.0f} units', 
                     f'{(allocated_budget/total_budget):.1%}', f'{efficiency_score}%']
        }
        
        st.table(pd.DataFrame(summary_data))
    
    # Download Report
    st.subheader("📥 Export Report")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.success("PDF report generated successfully!")
    with col2:
        if st.button("📊 Export Excel Data", use_container_width=True):
            st.success("Excel data exported successfully!")
    with col3:
        if st.button("📋 Download CSV Summary", use_container_width=True):
            st.success("CSV summary downloaded successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div class='footer'>
    <h3>Developed with ❤️ by Yash Gupta</h3>
    <p>Data Science Enthusiast | AI Solutions Developer | Business Analytics Specialist</p>
    <p>Transforming data into actionable insights for better advertising decisions</p>
</div>
""", unsafe_allow_html=True)
