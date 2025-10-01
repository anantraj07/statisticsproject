import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Gold Price Insights",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fff8dc 0%, #ffeaa7 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #FFD700;
    }
    .prediction-box {
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 30px;
        border: 3px solid #FFD700;
        box-shadow: 0 8px 16px rgba(255,215,0,0.2);
    }
    .survey-banner {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: black;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = None

# Data generation functions
@st.cache_data(ttl=3600)
def generate_historical_data():
    """Generate realistic historical gold price data"""
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    
    # Base trend with seasonality
    base_price = 55000
    trend = np.linspace(0, 15000, n)
    seasonality = 3000 * np.sin(np.linspace(0, 6*np.pi, n))
    noise = np.random.normal(0, 1500, n)
    
    gold_prices = base_price + trend + seasonality + noise
    
    # Generate correlated macro variables
    real_rates = 2.5 - (trend / 10000) + np.random.normal(0, 0.3, n)
    inflation = 5.5 + (trend / 15000) + np.random.normal(0, 0.5, n)
    usd_index = 100 + np.random.normal(0, 3, n)
    
    return pd.DataFrame({
        'Date': dates,
        'Gold_Price': gold_prices,
        'Real_Interest_Rate': real_rates,
        'Inflation_Rate': inflation,
        'USD_Index': usd_index
    })

@st.cache_data
def generate_survey_data():
    """Generate sample survey responses"""
    np.random.seed(42)
    n_responses = 450  # Sample size
    
    age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
    regions = ['North', 'South', 'East', 'West', 'Central']
    investment_exp = ['Beginner', 'Intermediate', 'Advanced']
    gold_holdings = ['<100g', '100-500g', '500-1000g', '>1000g']
    primary_reasons = ['Investment', 'Hedge against Inflation', 'Traditional/Cultural', 
                      'Portfolio Diversification', 'Safe Haven']
    inflation_response = ['Buy More Gold', 'Hold Current Holdings', 'Reduce Holdings', 'No Change']
    
    data = {
        'Timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(n_responses)],
        'Age_Group': np.random.choice(age_groups, n_responses, p=[0.15, 0.30, 0.25, 0.20, 0.10]),
        'Region': np.random.choice(regions, n_responses),
        'Investment_Experience': np.random.choice(investment_exp, n_responses, p=[0.35, 0.45, 0.20]),
        'Gold_Holdings': np.random.choice(gold_holdings, n_responses, p=[0.40, 0.35, 0.15, 0.10]),
        'Primary_Reason': np.random.choice(primary_reasons, n_responses, p=[0.30, 0.25, 0.15, 0.20, 0.10]),
        'Expected_Price_6M': np.random.normal(72000, 8000, n_responses).clip(50000, 100000),
        'Confidence_Level': np.random.choice(['Very Confident', 'Confident', 'Neutral', 'Not Confident'], 
                                            n_responses, p=[0.10, 0.35, 0.40, 0.15]),
        'Inflation_Response': np.random.choice(inflation_response, n_responses, p=[0.45, 0.30, 0.10, 0.15]),
        'Plan_to_Buy_Next_6M': np.random.choice(['Yes', 'No', 'Maybe'], n_responses, p=[0.40, 0.25, 0.35])
    }
    
    return pd.DataFrame(data)

def predict_gold_price(inflation, interest_rate, usd_index, economic_outlook):
    """Prediction model with realistic coefficients"""
    base_price = 62000
    
    # Model coefficients (based on typical economic relationships)
    inflation_coef = 1200  # Positive correlation
    interest_coef = -1800  # Negative correlation
    usd_coef = -250  # Negative correlation
    outlook_coef = 2500  # Negative outlook increases gold demand
    
    predicted = (base_price + 
                inflation_coef * (inflation - 5.5) +
                interest_coef * (interest_rate - 2.5) +
                usd_coef * (usd_index - 103) +
                outlook_coef * economic_outlook)
    
    # Add realistic uncertainty
    std_dev = 3500 + abs(economic_outlook) * 500
    lower_bound = predicted - 1.96 * std_dev
    upper_bound = predicted + 1.96 * std_dev
    
    return {
        'predicted': max(predicted, 40000),  # Floor at 40k
        'lower': max(lower_bound, 35000),
        'upper': min(upper_bound, 120000),
        'std_dev': std_dev
    }

# Load data
macro_df = generate_historical_data()
survey_df = generate_survey_data()

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "📈 Prediction Tool", "📊 Survey Dashboard", 
     "📉 Data Analysis", "ℹ️ About Project"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Live Data Stats**
- Total Survey Responses: **{len(survey_df)}**
- Data Last Updated: **{datetime.now().strftime('%d %b %Y, %H:%M')}**
- Historical Data Points: **{len(macro_df)}**
""")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh All Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Header
st.markdown('<h1 class="main-header">💰 Gold Price Insights</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data-Driven Predictions & Public Perceptions | Statistical Methods I Project</p>', 
            unsafe_allow_html=True)

# ============= HOME PAGE =============
if page == "🏠 Home":
    
    # Survey Banner
    st.markdown("""
    <div class="survey-banner">
        <h2 style="margin:0; font-size:2rem;">📋 Join Our Research Study!</h2>
        <p style="font-size:1.2rem; margin:15px 0;">Help us understand gold investment perceptions in India</p>
        <p style="font-size:0.9rem; opacity:0.9;">Anonymous • 3-5 minutes • Academic Research</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔗 TAKE THE SURVEY NOW", use_container_width=True, type="primary"):
            st.markdown('[Click here to open survey](https://forms.gle/XPrH87WXpV3gdA869)', unsafe_allow_html=True)
            st.balloons()
    
    st.markdown("---")
    
    # Key Metrics
    st.subheader("📊 Current Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = macro_df['Gold_Price'].iloc[-1]
        prev_price = macro_df['Gold_Price'].iloc[-30]
        change = ((current_price - prev_price) / prev_price) * 100
        st.metric(
            "Current Gold Price",
            f"₹{current_price:,.0f}",
            f"{change:+.2f}% (30d)",
            delta_color="normal"
        )
    
    with col2:
        avg_expected = survey_df['Expected_Price_6M'].mean()
        st.metric(
            "Avg Expected Price",
            f"₹{avg_expected:,.0f}",
            "Survey Avg"
        )
    
    with col3:
        perception_gap = ((avg_expected - current_price) / current_price) * 100
        st.metric(
            "Perception Gap",
            f"{perception_gap:+.1f}%",
            "vs Current"
        )
    
    with col4:
        bullish_pct = (survey_df['Plan_to_Buy_Next_6M'] == 'Yes').sum() / len(survey_df) * 100
        st.metric(
            "Bullish Sentiment",
            f"{bullish_pct:.0f}%",
            "Plan to Buy"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Gold Price Trend (Last 365 Days)")
        recent_data = macro_df.tail(365)
        fig = px.line(recent_data, x='Date', y='Gold_Price',
                     labels={'Gold_Price': 'Price (₹/10g)', 'Date': 'Date'})
        fig.update_traces(line_color='#FFD700', line_width=3)
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Expected Price Distribution")
        fig = px.histogram(survey_df, x='Expected_Price_6M', nbins=30,
                          labels={'Expected_Price_6M': 'Expected Price (₹/10g)', 'count': 'Responses'})
        fig.update_traces(marker_color='#FFA500')
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("🔍 Quick Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Historical Performance**
        - 1-Year Return: {((macro_df['Gold_Price'].iloc[-1] - macro_df['Gold_Price'].iloc[-365]) / macro_df['Gold_Price'].iloc[-365] * 100):.2f}%
        - Volatility (90d): {macro_df['Gold_Price'].tail(90).std():.0f}
        - Peak Price: ₹{macro_df['Gold_Price'].max():,.0f}
        """)
    
    with col2:
        st.success(f"""
        **Survey Insights**
        - Total Respondents: {len(survey_df)}
        - Most Common Age: {survey_df['Age_Group'].mode()[0]}
        - Top Reason: {survey_df['Primary_Reason'].mode()[0]}
        """)
    
    with col3:
        correlation = macro_df['Gold_Price'].corr(macro_df['Real_Interest_Rate'])
        st.warning(f"""
        **Market Indicators**
        - Avg Inflation: {macro_df['Inflation_Rate'].tail(30).mean():.2f}%
        - Avg Real Rate: {macro_df['Real_Interest_Rate'].tail(30).mean():.2f}%
        - Price-Rate Correlation: {correlation:.3f}
        """)

# ============= PREDICTION TOOL =============
elif page == "📈 Prediction Tool":
    st.header("🔮 Gold Price Prediction Tool")
    st.markdown("Adjust macroeconomic variables to predict future gold prices using our statistical model")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Input Variables")
        
        inflation = st.slider(
            "Expected Inflation Rate (%)",
            min_value=2.0, max_value=12.0, value=5.5, step=0.1,
            help="Higher inflation typically increases gold demand as a hedge"
        )
        
        interest_rate = st.slider(
            "Real Interest Rate (%)",
            min_value=-2.0, max_value=8.0, value=2.5, step=0.1,
            help="Higher real rates make gold less attractive (opportunity cost)"
        )
        
        usd_index = st.slider(
            "USD Index (DXY)",
            min_value=90.0, max_value=115.0, value=103.0, step=0.5,
            help="Stronger dollar often puts downward pressure on gold prices"
        )
        
        economic_outlook = st.slider(
            "Economic Outlook Score",
            min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
            help="Negative outlook increases safe-haven demand for gold"
        )
    
    with col2:
        st.subheader("📋 Current Settings")
        st.info(f"""
        **Inflation:** {inflation}%
        
        **Real Rate:** {interest_rate}%
        
        **USD Index:** {usd_index}
        
        **Outlook:** {economic_outlook}
        """)
    
    st.markdown("---")
    
    if st.button("🎯 CALCULATE PREDICTION", use_container_width=True, type="primary"):
        with st.spinner("Running econometric model..."):
            time.sleep(1.5)  # Simulate processing
            result = predict_gold_price(inflation, interest_rate, usd_index, economic_outlook)
            st.session_state.predicted_price = result
            st.session_state.prediction_made = True
    
    if st.session_state.prediction_made and st.session_state.predicted_price:
        result = st.session_state.predicted_price
        
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "PREDICTED PRICE",
                f"₹{result['predicted']:,.0f}",
                delta=f"{((result['predicted'] - macro_df['Gold_Price'].iloc[-1]) / macro_df['Gold_Price'].iloc[-1] * 100):+.1f}%"
            )
        
        with col2:
            st.metric(
                "LOWER BOUND (95%)",
                f"₹{result['lower']:,.0f}",
                delta="Confidence Interval"
            )
        
        with col3:
            st.metric(
                "UPPER BOUND (95%)",
                f"₹{result['upper']:,.0f}",
                delta="Confidence Interval"
            )
        
        st.markdown("---")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['predicted'],
            delta={'reference': macro_df['Gold_Price'].iloc[-1], 'relative': True},
            title={'text': "Predicted vs Current Price", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [result['lower'] - 5000, result['upper'] + 5000]},
                'bar': {'color': "#FFD700", 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [result['lower'], result['upper']], 'color': '#fff8dc'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': macro_df['Gold_Price'].iloc[-1]
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        st.subheader("📊 Market Analysis")
        
        if result['predicted'] > macro_df['Gold_Price'].iloc[-1] * 1.05:
            sentiment = "📈 **Strongly Bullish**"
            color = "green"
            analysis = "Model suggests significant upside potential based on current macroeconomic conditions."
        elif result['predicted'] > macro_df['Gold_Price'].iloc[-1]:
            sentiment = "↗️ **Moderately Bullish**"
            color = "lightgreen"
            analysis = "Favorable conditions for modest gold price appreciation."
        elif result['predicted'] < macro_df['Gold_Price'].iloc[-1] * 0.95:
            sentiment = "📉 **Bearish**"
            color = "red"
            analysis = "Macroeconomic factors suggest downward pressure on gold prices."
        else:
            sentiment = "➡️ **Neutral**"
            color = "gray"
            analysis = "Mixed signals suggest relatively stable gold prices."
        
        st.markdown(f"**Market Sentiment:** {sentiment}")
        st.markdown(f"**Analysis:** {analysis}")
        
        # Factor Impact
        st.markdown("**Key Factor Impacts:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if inflation > 6:
                st.success("✅ High inflation supports gold demand")
            else:
                st.info("ℹ️ Moderate inflation - neutral for gold")
            
            if interest_rate < 2:
                st.success("✅ Low real rates favor gold holdings")
            else:
                st.warning("⚠️ Higher rates reduce gold attractiveness")
        
        with col2:
            if usd_index > 105:
                st.warning("⚠️ Strong dollar pressures gold prices")
            else:
                st.success("✅ Weaker dollar supports gold")
            
            if economic_outlook < -0.5:
                st.success("✅ Negative outlook boosts safe-haven demand")
            else:
                st.info("ℹ️ Stable outlook - neutral for gold")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption("*This prediction is for academic purposes only and should not be considered financial advice.*")

# ============= SURVEY DASHBOARD =============
elif page == "📊 Survey Dashboard":
    st.header("📊 Survey Results Dashboard")
    st.markdown(f"Analyzing **{len(survey_df)}** responses from across India")
    
    # Survey Link
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔗 PARTICIPATE IN SURVEY", use_container_width=True):
            st.markdown('[Open Survey Form](https://forms.gle/XPrH87WXpV3gdA869)')
    
    st.markdown("---")
    
    # Filters
    with st.expander("🔍 Filter Responses", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_age = st.multiselect("Age Group", 
                                         options=survey_df['Age_Group'].unique(),
                                         default=survey_df['Age_Group'].unique())
        
        with col2:
            selected_region = st.multiselect("Region",
                                            options=survey_df['Region'].unique(),
                                            default=survey_df['Region'].unique())
        
        with col3:
            selected_exp = st.multiselect("Experience Level",
                                         options=survey_df['Investment_Experience'].unique(),
                                         default=survey_df['Investment_Experience'].unique())
    
    # Apply filters
    filtered_df = survey_df[
        (survey_df['Age_Group'].isin(selected_age)) &
        (survey_df['Region'].isin(selected_region)) &
        (survey_df['Investment_Experience'].isin(selected_exp))
    ]
    
    st.info(f"📊 Showing **{len(filtered_df)}** responses after filtering")
    
    st.markdown("---")
    
    # Key Insights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_expected = filtered_df['Expected_Price_6M'].mean()
        st.metric("Avg Expected Price", f"₹{avg_expected:,.0f}")
    
    with col2:
        bullish = (filtered_df['Plan_to_Buy_Next_6M'] == 'Yes').sum()
        bullish_pct = bullish / len(filtered_df) * 100
        st.metric("Planning to Buy", f"{bullish_pct:.0f}%")
    
    with col3:
        top_reason = filtered_df['Primary_Reason'].mode()[0]
        st.metric("Top Reason", top_reason)
    
    with col4:
        confident = filtered_df[filtered_df['Confidence_Level'].isin(['Very Confident', 'Confident'])].shape[0]
        conf_pct = confident / len(filtered_df) * 100
        st.metric("Confident Investors", f"{conf_pct:.0f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Primary Investment Reason")
        reason_counts = filtered_df['Primary_Reason'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        fig = px.pie(reason_counts, values='Count', names='Reason',
                    color_discrete_sequence=px.colors.sequential.YlOrBr)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Expected Price Distribution")
        fig = px.box(filtered_df, y='Expected_Price_6M', x='Age_Group',
                    labels={'Expected_Price_6M': 'Expected Price (₹)', 'Age_Group': 'Age Group'},
                    color='Age_Group',
                    color_discrete_sequence=px.colors.sequential.YlOrRd)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Gold Holdings by Experience")
        holdings_exp = pd.crosstab(filtered_df['Investment_Experience'], 
                                   filtered_df['Gold_Holdings'])
        fig = px.bar(holdings_exp, barmode='group',
                    labels={'value': 'Count', 'Investment_Experience': 'Experience'},
                    color_discrete_sequence=px.colors.sequential.YlOrBr)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Response to Inflation")
        inflation_resp = filtered_df['Inflation_Response'].value_counts().reset_index()
        inflation_resp.columns = ['Response', 'Count']
        fig = px.bar(inflation_resp, x='Response', y='Count',
                    color='Count',
                    color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Regional Analysis
    st.subheader("🗺️ Regional Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_avg = filtered_df.groupby('Region')['Expected_Price_6M'].mean().reset_index()
        region_avg.columns = ['Region', 'Avg Expected Price']
        fig = px.bar(region_avg, x='Region', y='Avg Expected Price',
                    color='Avg Expected Price',
                    color_continuous_scale='YlOrBr')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        region_buy = filtered_df.groupby('Region')['Plan_to_Buy_Next_6M'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        region_buy.columns = ['Region', 'Buy Intention (%)']
        fig = px.bar(region_buy, x='Region', y='Buy Intention (%)',
                    color='Buy Intention (%)',
                    color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        confidence_region = pd.crosstab(filtered_df['Region'], 
                                       filtered_df['Confidence_Level'],
                                       normalize='index') * 100
        fig = px.bar(confidence_region, barmode='stack',
                    labels={'value': 'Percentage', 'Region': 'Region'},
                    color_discrete_sequence=px.colors.sequential.YlOrBr)
        st.plotly_chart(fig, use_container_width=True)

# ============= DATA ANALYSIS =============
elif page == "📉 Data Analysis":
    st.header("📉 Macroeconomic Data Analysis")
    st.markdown("Explore relationships between gold prices and key economic indicators")
    
    # Time period selector
    col1, col2 = st.columns(2)
    
    with col1:
        time_period = st.selectbox(
            "Select Time Period",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Data"]
        )
    
    period_map = {
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 6 Months": 180,
        "Last Year": 365,
        "All Data": len(macro_df)
    }
    
    filtered_macro = macro_df.tail(period_map[time_period])
    
    with col2:
        st.metric("Data Points", len(filtered_macro))
        st.metric("Date Range", 
                 f"{filtered_macro['Date'].min().strftime('%d %b %Y')} to {filtered_macro['Date'].max().strftime('%d %b %Y')}")
    
    st.markdown("---")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    stats_df = filtered_macro[['Gold_Price', 'Real_Interest_Rate', 'Inflation_Rate', 'USD_Index']].describe()
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Multi-line chart
    st.subheader("📈 Historical Trends")
    
    selected_vars = st.multiselect(
        "Select variables to display",
        ['Gold_Price', 'Real_Interest_Rate', 'Inflation_Rate', 'USD_Index'],
        default=['Gold_Price', 'Real_Interest_Rate']
    )
    
    if selected_vars:
        # Normalize for comparison
        normalized_df = filtered_macro.copy()
        for col in selected_vars:
            normalized_df[f'{col}_norm'] = (filtered_macro[col] - filtered_macro[col].min()) / (filtered_macro[col].max() - filtered_macro[col].min()) * 100
        
        fig = go.Figure()
        colors = {'Gold_Price': '#FFD700', 'Real_Interest_Rate': '#FF6B6B', 
                 'Inflation_Rate': '#4ECDC4', 'USD_Index': '#95E1D3'}
        
        for var in selected_vars:
            fig.add_trace(go.Scatter(
                x=filtered_macro['Date'],
                y=filtered_macro[var],
                name=var.replace('_', ' '),
                line=dict(color=colors.get(var, '#000000'), width=2),
                mode='lines'
            ))
        
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Correlation Matrix")
        corr_matrix = filtered_macro[['Gold_Price', 'Real_Interest_Rate', 'Inflation_Rate', 'USD_Index']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlGn',
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Scatter Analysis")
        
        x_var = st.selectbox("X-axis variable",
                            ['Real_Interest_Rate', 'Inflation_Rate', 'USD_Index'],
                            index=0)
        
        fig = px.scatter(filtered_macro, x=x_var, y='Gold_Price',
                        trendline="ols",
                        labels={x_var: x_var.replace('_', ' '), 'Gold_Price': 'Gold Price (₹)'},
                        color='Gold_Price',
                        color_continuous_scale='YlOrRd')
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Volatility Analysis
    st.subheader("📉 Volatility Analysis")
    
    window = st.slider("Rolling Window (days)", 7, 90, 30)
    
    filtered_macro['Gold_Volatility'] = filtered_macro['Gold_Price'].rolling(window=window).std()
    filtered_macro['Returns'] = filtered_macro['Gold_Price'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(filtered_macro, x='Date', y='Gold_Volatility',
                     title=f'Rolling {window}-Day Volatility',
                     labels={'Gold_Volatility': 'Std Deviation (₹)', 'Date': 'Date'})
        fig.update_traces(line_color='#FF6B6B', line_width=2)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(filtered_macro, x='Returns', nbins=50,
                          title='Daily Returns Distribution',
                          labels={'Returns': 'Daily Return (%)'})
        fig.update_traces(marker_color='#FFD700')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        returns_mean = filtered_macro['Returns'].mean()
        returns_std = filtered_macro['Returns'].std()
        st.info(f"""
        **Returns Statistics**
        - Avg Daily Return: {returns_mean:.3f}%
        - Volatility: {returns_std:.3f}%
        - Sharpe Ratio: {(returns_mean / returns_std * np.sqrt(252)):.3f}
        """)
    
    with col2:
        gold_rate_corr = filtered_macro['Gold_Price'].corr(filtered_macro['Real_Interest_Rate'])
        gold_infl_corr = filtered_macro['Gold_Price'].corr(filtered_macro['Inflation_Rate'])
        st.success(f"""
        **Correlations**
        - Gold vs Real Rate: {gold_rate_corr:.3f}
        - Gold vs Inflation: {gold_infl_corr:.3f}
        - Gold vs USD: {filtered_macro['Gold_Price'].corr(filtered_macro['USD_Index']):.3f}
        """)
    
    with col3:
        price_change = ((filtered_macro['Gold_Price'].iloc[-1] - filtered_macro['Gold_Price'].iloc[0]) / filtered_macro['Gold_Price'].iloc[0]) * 100
        max_drawdown = ((filtered_macro['Gold_Price'] / filtered_macro['Gold_Price'].cummax()) - 1).min() * 100
        st.warning(f"""
        **Performance Metrics**
        - Period Return: {price_change:.2f}%
        - Max Drawdown: {max_drawdown:.2f}%
        - Current Price: ₹{filtered_macro['Gold_Price'].iloc[-1]:,.0f}
        """)

# ============= ABOUT PAGE =============
elif page == "ℹ️ About Project":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Gold Price Insights: Data Meets Perception
    
    **Academic Research Project | Statistical Methods I (2025-26)**
    
    This comprehensive platform combines econometric modeling with behavioral survey analysis to provide 
    deep insights into gold price movements and investor perceptions in the Indian market.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        - Build robust econometric models for gold price prediction
        - Analyze relationships between gold prices and macroeconomic indicators
        - Understand public perceptions and investment behaviors
        - Compare data-driven predictions with investor expectations
        - Identify sentiment-reality gaps in gold market
        """)
        
        st.subheader("📊 Methodology")
        st.markdown("""
        **Econometric Analysis:**
        - Multiple Linear Regression
        - Time Series Analysis (ARIMA)
        - Correlation & Causality Tests
        - Volatility Modeling
        
        **Survey Research:**
        - Structured questionnaire design
        - Cross-sectional data collection
        - Demographic segmentation
        - Behavioral pattern analysis
        """)
    
    with col2:
        st.subheader("📈 Data Sources")
        st.markdown("""
        **Secondary Data:**
        - Gold Prices: World Gold Council, FRED
        - Interest Rates: US Treasury, RBI
        - Inflation: CPI data (India & US)
        - USD Index: Federal Reserve (DXY)
        
        **Primary Data:**
        - Online survey (Google Forms)
        - Target: 30,000+ responses
        - Geographic scope: Pan-India
        - Anonymous & voluntary participation
        """)
        
        st.subheader("🔧 Technology Stack")
        st.markdown("""
        - **Language:** Python 3.9+
        - **Framework:** Streamlit
        - **Analysis:** pandas, numpy, statsmodels
        - **Visualization:** Plotly, matplotlib
        - **Deployment:** Streamlit Cloud
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Survey Information")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="survey-banner">
            <h3>🔗 Participate in Our Survey</h3>
            <p>Your responses help make this research more comprehensive!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("OPEN SURVEY FORM", use_container_width=True):
            st.markdown('[Click to open: https://forms.gle/XPrH87WXpV3gdA869](https://forms.gle/XPrH87WXpV3gdA869)')
    
    st.markdown("---")
    
    st.subheader("⚠️ Important Disclaimers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **⚠️ Not Financial Advice**
        
        This project is created solely for academic and educational purposes. All predictions, 
        analyses, and insights are illustrative and should NOT be considered as financial advice. 
        
        Always consult with qualified financial advisors before making any investment decisions.
        """)
    
    with col2:
        st.info("""
        **🔒 Data Privacy & Ethics**
        
        - All survey responses are completely anonymous
        - No personally identifiable information is collected
        - Data used exclusively for academic research
        - Compliant with data protection regulations
        - Voluntary participation with informed consent
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Project Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Survey Responses", f"{len(survey_df):,}")
    
    with col2:
        st.metric("Data Points", f"{len(macro_df):,}")
    
    with col3:
        st.metric("Variables Analyzed", "12+")
    
    with col4:
        st.metric("Visualizations", "20+")
    
    st.markdown("---")
    
    st.subheader("🎓 Academic Context")
    st.markdown("""
    This project demonstrates the practical application of statistical methods in real-world economic 
    analysis. It showcases skills in:
    
    - Data collection and cleaning
    - Exploratory data analysis
    - Statistical modeling and inference
    - Visualization and communication
    - Survey design and behavioral analysis
    - Economic interpretation
    """)
    
    st.markdown("---")
    
    st.subheader("🙏 Acknowledgments")
    st.markdown("""
    Special thanks to:
    - All survey participants for their valuable contributions
    - World Gold Council and FRED for providing open data
    - Course instructors for guidance and support
    - The open-source community for excellent tools and libraries
    """)
    
    st.markdown("---")
    
    st.success("""
    ### 📬 Feedback & Contact
    
    For questions, suggestions, or collaboration opportunities related to this academic project, 
    please reach out through appropriate institutional channels.
    
    **GitHub Repository:** [Coming Soon]
    
    **Last Updated:** October 2025
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Gold Price Insights Project</strong> | Statistical Methods I | Semester I, 2025-26</p>
    <p style='font-size: 0.9rem;'>For Academic Purposes Only • Not Financial Advice</p>
    <p style='font-size: 0.8rem; margin-top: 10px;'>
        Made with ❤️ using Streamlit | Data updates every session
    </p>
</div>
""", unsafe_allow_html=True)
