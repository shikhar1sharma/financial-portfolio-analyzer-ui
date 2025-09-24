"""
Complete Financial AI Assistant - MCP Host UI
Save this as: streamlit_app.py
"""
print("DEBUG: Starting imports...")
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List, Optional
import logging
import time
import re

print("DEBUG: All imports successful")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .status-good { 
        color: #28a745; 
        font-weight: bold;
    }
    
    .status-bad { 
        color: #dc3545; 
        font-weight: bold;
    }
    
    .status-warning { 
        color: #ffc107; 
        font-weight: bold;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        border-left: 4px solid #1f77b4;
        background: rgba(31, 119, 180, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .portfolio-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .market-overview {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

class MCPHost:
    """Advanced MCP Host for Financial AI System"""
    
    def __init__(self):
        print("DEBUG: MCPHost initializing...")
        # MCP Server URLs
        self.portfolio_server = os.getenv("PORTFOLIO_SERVER_URL", "http://localhost:8002")
        self.market_data_server = os.getenv("MARKET_DATA_SERVER_URL", "http://localhost:8001")
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        session_vars = {
            'chat_history': [],
            'portfolio_data': None,
            'market_data': None,
            'last_update': None,
            'connectivity': {},
            'selected_stocks': [],
            'user_preferences': {
                'auto_refresh': True,
                'notifications': True,
                'theme': 'professional'
            },
            'analysis_cache': {},
            'portfolio_alerts': []
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    def call_mcp_tool(self, server_url: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Enhanced MCP tool calling with retry logic and caching"""
        cache_key = f"{server_url}_{tool_name}_{str(sorted(kwargs.items()))}"
        
        # Check cache for non-critical data
        if tool_name in ['get_market_overview'] and cache_key in st.session_state.analysis_cache:
            cache_data = st.session_state.analysis_cache[cache_key]
            if datetime.now() - cache_data['timestamp'] < timedelta(minutes=5):
                return cache_data['data']
        
        try:
            # Determine HTTP method
            if tool_name in ["get_portfolio_overview", "get_asset_allocation", "get_portfolio_metrics"]:
                response = requests.get(f"{server_url}/tools/{tool_name}", timeout=30)
            elif tool_name == "get_market_overview":
                response = requests.get(f"{server_url}/tools/{tool_name}", timeout=30)
            else:
                response = requests.post(f"{server_url}/tools/{tool_name}", json=kwargs, timeout=30)
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful responses
            if tool_name in ['get_market_overview']:
                st.session_state.analysis_cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
            
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling {tool_name}: {e}")
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {e}")
            return {"error": str(e)}
    
    def test_server_connectivity(self) -> Dict[str, bool]:
        """Enhanced connectivity testing with detailed status"""
        servers = {
            "Portfolio Server": self.portfolio_server,
            "Market Data Server": self.market_data_server
        }
        
        connectivity = {}
        for name, url in servers.items():
            try:
                start_time = time.time()
                response = requests.get(f"{url}/health", timeout=10)
                response_time = time.time() - start_time
                
                connectivity[name] = {
                    'status': response.status_code == 200,
                    'response_time': round(response_time * 1000, 2),  # ms
                    'last_check': datetime.now().isoformat()
                }
            except Exception as e:
                connectivity[name] = {
                    'status': False,
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
        
        return connectivity

def main():
    """Main Streamlit application with enhanced features"""
    print("DEBUG: Main function called")
    
    # Initialize MCP Host
    mcp_host = MCPHost()
    
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">💰 Financial AI Assistant</h1>', unsafe_allow_html=True)
    # st.markdown(
    #     '<p style="text-align: center; color: #666; font-style: italic; font-size: 1.2rem;">'
    #     'Powered by MCP (Model Context Protocol) • Advanced Portfolio Analytics</p>',
    #     unsafe_allow_html=True
    # )
    
    # Create main layout
    create_sidebar(mcp_host)
    create_main_content(mcp_host)

def create_sidebar(mcp_host: MCPHost):
    """Enhanced sidebar with comprehensive controls"""
    with st.sidebar:
        # Server Status Section
        # st.markdown("### 🔧 System Status")
        #
        # # Auto-refresh toggle
        # auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.user_preferences['auto_refresh'])
        # st.session_state.user_preferences['auto_refresh'] = auto_refresh
        #
        # # Manual refresh or auto-refresh logic
        # if st.button("🔄 Refresh Status", use_container_width=True) or auto_refresh:
        #     if (st.session_state.last_update is None or
        #         (datetime.now() - st.session_state.last_update).seconds > 30):
        #
        #         with st.spinner("Checking connectivity..."):
        #             connectivity = mcp_host.test_server_connectivity()
        #             st.session_state.connectivity = connectivity
        #             st.session_state.last_update = datetime.now()
        #
        # # Display enhanced server status
        # display_server_status()
        #
        # st.divider()
        
        # Quick Actions Section
        st.markdown("### 📊 Quick Actions")
        create_quick_actions(mcp_host)
        
        st.divider()
        
        # Portfolio Quick View
        st.markdown("### 💼 Portfolio")
        display_portfolio_sidebar(mcp_host)
        
        st.divider()
        
        # Market Quick View
        st.markdown("### 📈 Market")
        display_market_sidebar(mcp_host)
        
        # st.divider()
        
        # Settings Section
        # create_settings_section()

def display_server_status():
    """Display enhanced server status information"""
    connectivity = st.session_state.connectivity
    
    for server_name, status_info in connectivity.items():
        if isinstance(status_info, dict) and 'status' in status_info:
            status = status_info['status']
            response_time = status_info.get('response_time', 0)
            
            if status:
                st.markdown(
                    f'<div class="status-good">✅ {server_name}</div>'
                    f'<small style="color: #666;">Response: {response_time}ms</small>', 
                    unsafe_allow_html=True
                )
            else:
                error = status_info.get('error', 'Unknown error')
                st.markdown(f'<div class="status-bad">❌ {server_name}</div>', unsafe_allow_html=True)
                st.caption(f"Error: {error}")
        else:
            # Fallback for simple boolean status
            if status_info:
                st.markdown(f'<div class="status-good">✅ {server_name}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-bad">❌ {server_name}</div>', unsafe_allow_html=True)

def create_quick_actions(mcp_host: MCPHost):
    """Create enhanced quick action buttons"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Portfolio", use_container_width=True):
            refresh_portfolio_data(mcp_host)
        
        # if st.button("🔍 Analysis", use_container_width=True):
        #     st.session_state.show_analysis = True
    
    with col2:
        if st.button("📰 Market", use_container_width=True):
            refresh_market_data(mcp_host)
        
        # if st.button("➕ Position", use_container_width=True):
        #     st.session_state.show_add_position = True
    
    # Advanced actions
    # if st.button("📊 Dashboard", use_container_width=True):
    #     st.session_state.show_dashboard = True
    #
    # if st.button("⚠️ Risk Report", use_container_width=True):
    #     st.session_state.show_risk_analysis = True

def display_portfolio_sidebar(mcp_host: MCPHost):
    """Enhanced portfolio sidebar display"""
    if st.session_state.portfolio_data is None:
        portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
        if "error" not in portfolio_data:
            st.session_state.portfolio_data = portfolio_data
    
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data and portfolio_data.get("status") == "success":
        summary = portfolio_data.get("portfolio_summary", {})
        
        # Portfolio metrics
        total_value = summary.get('total_market_value', 0)
        total_return = summary.get('total_return_percent', 0)
        total_pnl = summary.get('total_unrealized_pnl', 0)
        
        # Color-coded display
        return_color = "green" if total_return >= 0 else "red"
        pnl_color = "green" if total_pnl >= 0 else "red"
        
        st.markdown(f"""
        <div class="sidebar-card">
            <h4>💰 ${total_value:,.2f}</h4>
            <p style="color: {return_color}; font-weight: bold;">
                {total_return:+.2f}% (${total_pnl:,.2f})
            </p>
            <p style="color: #666; font-size: 0.9rem;">
                {summary.get('number_of_positions', 0)} positions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top holdings
        positions = portfolio_data.get("positions", [])
        if positions:
            st.caption("🔝 Top Holdings:")
            for i, pos in enumerate(positions[:3]):
                pnl_pct = pos.get('pnl_percent', 0)
                emoji = "📈" if pnl_pct > 0 else "📉" if pnl_pct < 0 else "➖"
                allocation = pos.get('allocation_percent', 0)
                st.caption(f"{emoji} {pos['symbol']}: {allocation:.1f}% ({pnl_pct:+.1f}%)")
    else:
        st.info("💼 No portfolio data\nClick 'Portfolio' to load")

def display_market_sidebar(mcp_host: MCPHost):
    """Enhanced market sidebar display"""
    try:
        market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
        
        if "error" not in market_data:
            indices = market_data.get("indices", {})
            
            # Display major indices
            for name, data in list(indices.items())[:3]:
                change_pct = data.get('change_percent', 0)
                price = data.get('current_price', 0)
                
                color = "green" if change_pct > 0 else "red" if change_pct < 0 else "gray"
                emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➖"
                
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.2rem 0; border-left: 3px solid {color};">
                    <strong>{name}</strong><br>
                    <span style="color: {color};">{emoji} {price:.2f} ({change_pct:+.2f}%)</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("📊 Market data unavailable")
            
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")

def create_settings_section():
    """Create settings and preferences section"""
    with st.expander("⚙️ Settings"):
        st.markdown("**Preferences**")
        
        # Notification settings
        notifications = st.checkbox(
            "🔔 Enable Notifications", 
            value=st.session_state.user_preferences['notifications']
        )
        st.session_state.user_preferences['notifications'] = notifications
        
        # Theme selection
        theme = st.selectbox(
            "🎨 Theme", 
            ["Professional", "Dark", "Light"],
            index=0
        )
        st.session_state.user_preferences['theme'] = theme.lower()
        
        # Data refresh interval
        refresh_interval = st.selectbox(
            "🔄 Refresh Interval",
            ["30 seconds", "1 minute", "5 minutes", "Manual"],
            index=1
        )
        
        st.markdown("**System Info**")
        st.caption(f"Portfolio Server: {os.getenv('PORTFOLIO_SERVER_URL', 'localhost:8002')}")
        st.caption(f"Market Server: {os.getenv('MARKET_DATA_SERVER_URL', 'localhost:8001')}")
        st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'Never'}")

def create_main_content(mcp_host: MCPHost):
    """Create the main content area with tabs"""
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Assistant", 
        "📊 Dashboard", 
        "📈 Portfolio", 
        "🔍 Analysis", 
        "📰 Market"
    ])
    
    with tab1:
        create_ai_chat_interface(mcp_host)
    
    with tab2:
        create_dashboard_view(mcp_host)
    
    with tab3:
        create_portfolio_view(mcp_host)
    
    with tab4:
        create_analysis_view(mcp_host)
    
    with tab5:
        create_market_view(mcp_host)

def create_ai_chat_interface(mcp_host: MCPHost):
    """Enhanced AI chat interface with contextual awareness"""
    st.markdown("### 💬 Chat with Your Financial AI Assistant")
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Add action buttons for assistant responses
                if message["role"] == "assistant" and i == len(st.session_state.chat_history) - 1:
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("👍", key=f"like_{i}"):
                            st.success("Feedback recorded!")
                    with col2:
                        if st.button("📋", key=f"copy_{i}"):
                            st.info("Response copied!")
    
    # Enhanced chat input with suggestions
    # create_chat_suggestions()
    
    # Main chat input
    if prompt := st.chat_input("Ask about your portfolio, stocks, market analysis, or investment strategies..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and respond
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = process_enhanced_user_query(mcp_host, prompt)
                st.markdown(response)
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

def create_chat_suggestions():
    """Create quick suggestion buttons for common queries"""
    # st.markdown("**💡 Quick Questions:**")
    
    suggestions = [
        "Show my portfolio performance",
        "Analyze AAPL stock",
        "What's the market doing today?",
        "Should I buy or sell?",
        "Risk assessment",
        "Portfolio optimization"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                # Trigger the query
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        response = process_enhanced_user_query(MCPHost(), suggestion)
                        st.markdown(response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

def create_dashboard_view(mcp_host: MCPHost):
    """Comprehensive dashboard view"""
    st.markdown("### 📊 Financial Dashboard")
    
    # Refresh data
    if st.button("🔄 Refresh Dashboard"):
        refresh_all_data(mcp_host)
    
    # Create dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Portfolio metrics
    portfolio_data = st.session_state.portfolio_data
    if portfolio_data and portfolio_data.get("status") == "success":
        summary = portfolio_data.get("portfolio_summary", {})
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${summary.get('total_market_value', 0):,.2f}",
                f"{summary.get('total_return_percent', 0):+.2f}%"
            )
        
        with col2:
            st.metric(
                "Total P&L",
                f"${summary.get('total_unrealized_pnl', 0):,.2f}",
                f"{summary.get('total_return_percent', 0):+.2f}%"
            )
        
        with col3:
            st.metric(
                "Positions",
                summary.get('number_of_positions', 0)
            )
    
    # Market overview metric
    market_data = st.session_state.market_data
    if market_data and "indices" in market_data:
        sp500 = market_data["indices"].get("S&P 500", {})
        with col4:
            st.metric(
                "S&P 500",
                f"{sp500.get('current_price', 0):.2f}",
                f"{sp500.get('change_percent', 0):+.2f}%"
            )
    
    st.divider()
    
    # Charts section
    create_dashboard_charts(mcp_host)

def create_dashboard_charts(mcp_host: MCPHost):
    """Create comprehensive dashboard charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Portfolio Allocation")
        create_portfolio_allocation_chart()
    
    with col2:
        st.markdown("#### 📊 Market Performance")
        create_market_performance_chart()
    
    # Performance over time chart
    st.markdown("#### 📈 Portfolio Performance Over Time")
    create_performance_timeline_chart()

def create_portfolio_allocation_chart():
    """Create portfolio allocation pie chart"""
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data and portfolio_data.get("status") == "success":
        positions = portfolio_data.get("positions", [])
        
        if positions:
            # Prepare data
            symbols = [pos['symbol'] for pos in positions]
            allocations = [pos.get('allocation_percent', 0) for pos in positions]
            values = [pos.get('market_value', 0) for pos in positions]
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=allocations,
                hole=.3,
                textinfo='label+percent',
                textposition='outside',
                marker_colors=px.colors.qualitative.Set3
            )])
            
            fig.update_layout(
                showlegend=True,
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio positions to display")
    else:
        st.info("Portfolio data not available")

def create_market_performance_chart():
    """Create market indices performance chart"""
    market_data = st.session_state.market_data
    
    if market_data and "indices" in market_data:
        indices = market_data["indices"]
        
        # Prepare data
        names = list(indices.keys())
        changes = [data.get("change_percent", 0) for data in indices.values()]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=changes,
                marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in changes],
                text=[f"{x:+.2f}%" for x in changes],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Market Indices Performance",
            yaxis_title="Change %",
            xaxis_title="Index",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market data not available")

def create_performance_timeline_chart():
    """Create a simulated performance timeline"""
    # This would ideally use historical data from your backend
    # For now, creating a sample timeline
    
    import numpy as np
    
    # Sample data - replace with actual historical data from your backend
    dates = pd.date_range(start='2024-01-01', end='2024-08-28', freq='D')
    np.random.seed(42)  # For consistent demo data
    
    # Simulate portfolio performance
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
    portfolio_values = [10000]  # Starting value
    
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # Create DataFrame
    df = pd.DataFrame({
    'Date': dates[:len(portfolio_values)],
    'Portfolio Value': portfolio_values[:len(dates)]
})
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Portfolio Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("📝 Note: This is demo data. Integrate with your backend for actual historical performance.")

def create_portfolio_view(mcp_host: MCPHost):
    """Detailed portfolio management view"""
    st.markdown("### 📈 Portfolio Management")
    
    # Portfolio actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Portfolio", use_container_width=True):
            refresh_portfolio_data(mcp_host)
    
    with col2:
        if st.button("➕ Add Position", use_container_width=True):
            st.session_state.show_add_position = True
    
    with col3:
        if st.button("📊 Export Data", use_container_width=True):
            export_portfolio_data()
    
    # Show add position form if requested
    if st.session_state.get('show_add_position'):
        create_add_position_form(mcp_host)
        if st.button("❌ Cancel", key="cancel_add_position"):
            st.session_state.show_add_position = False
            st.rerun()
    
    st.divider()
    
    # Portfolio overview
    display_detailed_portfolio(mcp_host)

def create_add_position_form(mcp_host: MCPHost):
    """Enhanced add position form"""
    st.markdown("#### ➕ Add New Position")
    
    with st.form("add_position_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input(
                "Stock Symbol*",
                placeholder="e.g., AAPL, GOOGL, MSFT",
                help="Enter the stock ticker symbol"
            ).upper()
        
        with col2:
            quantity = st.number_input(
                "Quantity*",
                min_value=1,
                value=100,
                step=1,
                help="Number of shares to add"
            )
        
        with col3:
            purchase_price = st.number_input(
                "Purchase Price*",
                min_value=0.01,
                value=100.00,
                step=0.01,
                format="%.2f",
                help="Price per share when purchased"
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            purchase_date = st.date_input(
                "Purchase Date",
                value=datetime.now().date(),
                help="Date when the position was purchased"
            )
        
        with col5:
            # Get current price for validation
            if symbol and len(symbol) > 0:
                current_data = mcp_host.call_mcp_tool(
                    mcp_host.market_data_server,
                    "get_stock_price",
                    symbol=symbol,
                    period="1d"
                )
                
                if "error" not in current_data:
                    current_price = current_data.get('current_price', 0)
                    st.metric("Current Price", f"${current_price:.2f}")
                    
                    if current_price > 0:
                        unrealized_pnl = (current_price - purchase_price) * quantity
                        pnl_percent = ((current_price - purchase_price) / purchase_price) * 100
                        
                        if unrealized_pnl >= 0:
                            st.success(f"Projected P&L: +${unrealized_pnl:.2f} ({pnl_percent:+.2f}%)")
                        else:
                            st.error(f"Projected P&L: ${unrealized_pnl:.2f} ({pnl_percent:+.2f}%)")
        
        # Form submission
        submitted = st.form_submit_button("🔘 Add Position", use_container_width=True)
        
        if submitted:
            if not symbol:
                st.error("❌ Please enter a stock symbol")
            elif quantity <= 0:
                st.error("❌ Quantity must be greater than 0")
            elif purchase_price <= 0:
                st.error("❌ Purchase price must be greater than 0")
            else:
                add_position_to_portfolio(mcp_host, symbol, quantity, purchase_price, purchase_date)

def add_position_to_portfolio(mcp_host: MCPHost, symbol: str, quantity: int, purchase_price: float, purchase_date):
    """Add position to portfolio with enhanced feedback"""
    with st.spinner(f"Adding {quantity} shares of {symbol}..."):
        result = mcp_host.call_mcp_tool(
            mcp_host.portfolio_server,
            "add_stock_position",
            symbol=symbol,
            quantity=quantity,
            purchase_price=purchase_price,
            purchase_date=purchase_date.isoformat()
        )
        
        if "error" in result:
            st.error(f"❌ Error adding position: {result['error']}")
        elif result.get("status") == "success":
            st.success(f"✅ Successfully added {quantity} shares of {symbol}!")
            
            # Show detailed results
            with st.expander("📊 Position Details"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Market Value", f"${result.get('market_value', 0):.2f}")
                
                with col2:
                    unrealized_pnl = result.get('unrealized_pnl', 0)
                    st.metric("Unrealized P&L", f"${unrealized_pnl:.2f}")
                
                with col3:
                    action = result.get('action', 'added').title()
                    st.info(f"Action: {action}")
            
            # Clear cache and refresh
            st.session_state.portfolio_data = None
            st.session_state.show_add_position = False
            
            # Add to chat history
            chat_message = f"Added {quantity} shares of {symbol} at ${purchase_price:.2f}"
            st.session_state.chat_history.append({"role": "user", "content": chat_message})
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"✅ Position added successfully! Current market value: ${result.get('market_value', 0):.2f}"
            })
            
            time.sleep(2)  # Brief pause to show success message
            st.rerun()
        else:
            st.error("❌ Failed to add position")

def display_detailed_portfolio(mcp_host: MCPHost):
    """Display detailed portfolio information"""
    portfolio_data = st.session_state.portfolio_data
    
    if not portfolio_data:
        portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
        if "error" not in portfolio_data:
            st.session_state.portfolio_data = portfolio_data
    
    if portfolio_data and portfolio_data.get("status") == "success":
        summary = portfolio_data.get("portfolio_summary", {})
        positions = portfolio_data.get("positions", [])
        
        # Portfolio summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Value",
                f"${summary.get('total_market_value', 0):,.2f}",
                f"{summary.get('total_return_percent', 0):+.2f}%"
            )
        
        with col2:
            st.metric(
                "Cost Basis",
                f"${summary.get('total_cost_basis', 0):,.2f}"
            )
        
        with col3:
            pnl = summary.get('total_unrealized_pnl', 0)
            st.metric(
                "Unrealized P&L",
                f"${pnl:,.2f}",
                f"{summary.get('total_return_percent', 0):+.2f}%"
            )
        
        with col4:
            st.metric(
                "Positions",
                summary.get('number_of_positions', 0)
            )
        
        st.divider()
        
        # Positions table
        if positions:
            st.markdown("#### 📋 Current Positions")
            
            # Create DataFrame for better display
            df_positions = pd.DataFrame(positions)
            
            # Format columns
            if not df_positions.empty:
                df_display = df_positions.copy()
                df_display['current_price'] = df_display['current_price'].apply(lambda x: f"${x:.2f}")
                df_display['purchase_price'] = df_display['purchase_price'].apply(lambda x: f"${x:.2f}")
                df_display['market_value'] = df_display['market_value'].apply(lambda x: f"${x:,.2f}")
                df_display['cost_basis'] = df_display['cost_basis'].apply(lambda x: f"${x:,.2f}")
                df_display['unrealized_pnl'] = df_display['unrealized_pnl'].apply(lambda x: f"${x:,.2f}")
                df_display['pnl_percent'] = df_display['pnl_percent'].apply(lambda x: f"{x:+.2f}%")
                df_display['allocation_percent'] = df_display['allocation_percent'].apply(lambda x: f"{x:.1f}%")
                
                # Rename columns for display
                df_display = df_display.rename(columns={
                    'symbol': 'Symbol',
                    'quantity': 'Quantity',
                    'current_price': 'Current Price',
                    'purchase_price': 'Purchase Price',
                    'market_value': 'Market Value',
                    'cost_basis': 'Cost Basis',
                    'unrealized_pnl': 'P&L',
                    'pnl_percent': 'P&L %',
                    'allocation_percent': 'Allocation %',
                    'sector': 'Sector'
                })
                
                # Select columns to display
                display_columns = ['Symbol', 'Quantity', 'Current Price', 'Market Value', 'P&L', 'P&L %', 'Allocation %', 'Sector']
                df_display = df_display[display_columns]
                
                st.dataframe(df_display, use_container_width=True)
        else:
            st.info("📝 No positions in portfolio. Add some positions to get started!")
    else:
        st.error("❌ Unable to load portfolio data")

def create_analysis_view(mcp_host: MCPHost):
    """Comprehensive stock analysis view"""
    st.markdown("### 🔍 Stock Analysis & Research")
    
    # Stock analysis form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_symbol = st.text_input(
            "Enter Stock Symbol for Analysis:",
            placeholder="e.g., AAPL, GOOGL, MSFT, TSLA",
            help="Enter any valid stock ticker symbol"
        ).upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze Stock", disabled=not analysis_symbol, use_container_width=True)
    
    # Analysis results
    if analyze_button and analysis_symbol:
        display_comprehensive_stock_analysis(mcp_host, analysis_symbol)
    
    st.divider()
    
    # Recent analysis history
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        st.markdown("#### 📚 Recent Analysis")
        
        for i, analysis in enumerate(st.session_state.analysis_history[-5:]):  # Show last 5
            with st.expander(f"{analysis['symbol']} - {analysis['timestamp']}"):
                st.markdown(analysis['summary'])

def display_comprehensive_stock_analysis(mcp_host: MCPHost, symbol: str):
    """Display comprehensive stock analysis"""
    with st.spinner(f"Analyzing {symbol}..."):
        # Get stock data
        stock_data = mcp_host.call_mcp_tool(
            mcp_host.market_data_server,
            "get_stock_price",
            symbol=symbol,
            period="1y"
        )
        
        # Get technical analysis
        technical_data = mcp_host.call_mcp_tool(
            mcp_host.market_data_server,
            "get_technical_analysis",
            symbol=symbol
        )
        
        if "error" in stock_data:
            st.error(f"❌ Error analyzing {symbol}: {stock_data['error']}")
            return
        
        # Display analysis
        st.markdown(f"## 📊 {symbol} - {stock_data.get('company_name', 'Company Analysis')}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data.get('current_price', 0)
            price_change = stock_data.get('price_change', 0)
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"${price_change:+.2f}"
            )
        
        with col2:
            percent_change = stock_data.get('percent_change_pct', 0)
            st.metric(
                "Daily Change",
                f"{percent_change:+.2f}%"
            )
        
        with col3:
            volume = stock_data.get('volume', 0)
            st.metric(
                "Volume",
                f"{volume:,}"
            )
        
        with col4:
            market_cap = stock_data.get('market_cap', 0)
            if market_cap > 1e9:
                market_cap_display = f"${market_cap/1e9:.2f}B"
            elif market_cap > 1e6:
                market_cap_display = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_display = f"${market_cap:,.0f}"
            
            st.metric("Market Cap", market_cap_display)
        
        st.divider()
        
        # Detailed information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Key Metrics")
            metrics_data = {
                "52W High": f"${stock_data.get('high_52w', 0):.2f}",
                "52W Low": f"${stock_data.get('low_52w', 0):.2f}",
                "P/E Ratio": f"{stock_data.get('pe_ratio', 0):.2f}",
                "Beta": f"{stock_data.get('beta', 0):.2f}",
                "Dividend Yield": f"{stock_data.get('dividend_yield', 0)*100:.2f}%",
                "Sector": stock_data.get('sector', 'N/A'),
                "Industry": stock_data.get('industry', 'N/A')
            }
            
            for metric, value in metrics_data.items():
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.write(f"**{metric}:**")
                with col_b:
                    st.write(value)
        
        with col2:
            if "error" not in technical_data:
                st.markdown("#### 📊 Technical Analysis")
                
                ma = technical_data.get('moving_averages', {})
                rsi = technical_data.get('rsi')
                
                # Technical indicators
                tech_data = {
                    "RSI": f"{rsi:.2f}" if rsi else "N/A",
                    "MA20": f"${ma.get('ma_20', 0):.2f}" if ma.get('ma_20') else "N/A",
                    "MA50": f"${ma.get('ma_50', 0):.2f}" if ma.get('ma_50') else "N/A",
                    "MA200": f"${ma.get('ma_200', 0):.2f}" if ma.get('ma_200') else "N/A"
                }
                
                for indicator, value in tech_data.items():
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.write(f"**{indicator}:**")
                    with col_b:
                        st.write(value)
                
                # RSI interpretation
                if rsi:
                    if rsi > 70:
                        st.warning("⚠️ RSI indicates overbought conditions")
                    elif rsi < 30:
                        st.info("💡 RSI indicates oversold conditions")
                    else:
                        st.success("✅ RSI is in normal range")
        
        # Trading signals
        if "error" not in technical_data:
            signals = technical_data.get('signals', [])
            if signals:
                st.markdown("#### 🎯 Trading Signals")
                for signal in signals:
                    if "bullish" in signal.lower() or "buy" in signal.lower():
                        st.success(f"🟢 {signal}")
                    elif "bearish" in signal.lower() or "sell" in signal.lower():
                        st.error(f"🔴 {signal}")
                    else:
                        st.info(f"⚠️ {signal}")
        
        # Save to analysis history
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        analysis_summary = f"**{symbol}**: ${current_price:.2f} ({percent_change:+.2f}%) - {stock_data.get('sector', 'N/A')}"
        st.session_state.analysis_history.append({
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'summary': analysis_summary
        })
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": f"Analyze {symbol}"})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"📊 **{symbol} Analysis Complete**\n\n{analysis_summary}\n\nKey points:\n- Current Price: ${current_price:.2f}\n- Daily Change: {percent_change:+.2f}%\n- Market Cap: {market_cap_display}\n- RSI: {rsi:.2f if rsi else 'N/A'}"
        })

def create_market_view(mcp_host: MCPHost):
    """Comprehensive market overview"""
    st.markdown("### 📰 Market Overview & News")
    
    # Refresh market data
    if st.button("🔄 Refresh Market Data"):
        refresh_market_data(mcp_host)
    
    # Market indices overview
    display_market_indices(mcp_host)
    
    st.divider()
    
    # Market sentiment section
    st.markdown("#### 📊 Market Sentiment")
    display_market_sentiment()

def display_market_indices(mcp_host: MCPHost):
    """Display detailed market indices information"""
    market_data = st.session_state.market_data
    
    if not market_data:
        market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
        if "error" not in market_data:
            st.session_state.market_data = market_data
    
    if market_data and "indices" in market_data:
        indices = market_data["indices"]
        
        # Create metrics display
        indices_list = list(indices.items())
        
        # Display in rows of 3
        for i in range(0, len(indices_list), 3):
            cols = st.columns(3)
            
            for j, (name, data) in enumerate(indices_list[i:i+3]):
                with cols[j]:
                    current_price = data.get('current_price', 0)
                    change = data.get('change', 0)
                    change_percent = data.get('change_percent', 0)
                    
                    st.metric(
                        name,
                        f"{current_price:.2f}",
                        f"{change:+.2f} ({change_percent:+.2f}%)"
                    )
        
        # Market overview chart
        st.markdown("#### 📈 Market Performance Chart")
        names = list(indices.keys())
        changes = [data.get("change_percent", 0) for data in indices.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=changes,
                marker_color=['#00ff00' if x > 0 else '#ff0000' if x < 0 else '#gray' for x in changes],
                text=[f"{x:+.2f}%" for x in changes],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Market Indices Performance Today",
            yaxis_title="Change %",
            xaxis_title="Index",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("❌ Unable to load market data")

def display_market_sentiment():
    """Display market sentiment analysis"""
    # This would integrate with news APIs in a real implementation
    st.info("📰 Market sentiment analysis coming soon! This would integrate with news APIs to provide real-time market sentiment.")
    
    # Sample sentiment data for demo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Market Sentiment", "Neutral", "📊")
    
    with col2:
        st.metric("Fear & Greed Index", "45", "😐")
    
    with col3:
        st.metric("VIX Level", "18.5", "+2.1%")

# Enhanced query processing functions
def process_enhanced_user_query(mcp_host: MCPHost, query: str) -> str:
    """Enhanced query processing with context awareness"""
    query_lower = query.lower()
    
    try:
        # Portfolio queries
        if any(word in query_lower for word in ['portfolio', 'positions', 'holdings', 'my stocks']):
            return handle_portfolio_query(mcp_host)
        
        # Market queries
        elif any(word in query_lower for word in ['market', 'indices', 'sp500', 'nasdaq', 'dow']):
            return handle_market_query(mcp_host)
        
        # Stock analysis queries
        elif 'analyze' in query_lower or 'analysis' in query_lower:
            return handle_analysis_query(mcp_host, query)
        
        # Add position queries
        elif 'add' in query_lower and any(word in query_lower for word in ['position', 'stock', 'buy', 'shares']):
            return handle_add_position_query(mcp_host, query)
        
        # Performance queries
        elif any(word in query_lower for word in ['performance', 'return', 'profit', 'loss']):
            return handle_performance_query(mcp_host)
        
        # Risk queries
        elif any(word in query_lower for word in ['risk', 'volatility', 'beta']):
            return handle_risk_query(mcp_host)
        
        # General help
        else:
            return get_help_response()
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"❌ I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or check if the servers are responding properly."

def handle_portfolio_query(mcp_host: MCPHost) -> str:
    """Handle portfolio-related queries"""
    portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
    
    if "error" in portfolio_data:
        return f"❌ I couldn't retrieve your portfolio data: {portfolio_data['error']}\n\nPlease check if the portfolio server is running properly."
    
    if portfolio_data.get("status") == "success":
        summary = portfolio_data.get("portfolio_summary", {})
        positions = portfolio_data.get("positions", [])
        
        # Update session state
        st.session_state.portfolio_data = portfolio_data
        
        response = f"""
## 📊 Your Portfolio Overview

**💰 Portfolio Value:** ${summary.get('total_market_value', 0):,.2f}
**📈 Total P&L:** ${summary.get('total_unrealized_pnl', 0):,.2f} ({summary.get('total_return_percent', 0):+.2f}%)
**📋 Number of Positions:** {summary.get('number_of_positions', 0)}
**💼 Cost Basis:** ${summary.get('total_cost_basis', 0):,.2f}

### 🏆 Top Holdings:
"""
        for i, pos in enumerate(positions[:5]):
            emoji = "📈" if pos['pnl_percent'] > 0 else "📉" if pos['pnl_percent'] < 0 else "➖"
            response += f"{i+1}. {emoji} **{pos['symbol']}**: {pos['quantity']} shares @ ${pos['current_price']:.2f}\n"
            response += f"   💵 Value: ${pos['market_value']:,.2f} | P&L: {pos['pnl_percent']:+.2f}% | Allocation: {pos.get('allocation_percent', 0):.1f}%\n\n"
        
        if len(positions) > 5:
            response += f"*...and {len(positions) - 5} more positions*\n\n"
        
        # Add performance insight
        if summary.get('total_return_percent', 0) > 0:
            response += "🎉 **Great job!** Your portfolio is performing well with positive returns!"
        elif summary.get('total_return_percent', 0) < -5:
            response += "⚠️ Your portfolio is down significantly. Consider reviewing your positions or adding some diversification."
        else:
            response += "📊 Your portfolio performance is relatively stable."
        
        return response
    else:
        return "📝 **No portfolio data found.**\n\nIt looks like you don't have any positions yet. Would you like me to help you add your first position? Just say something like:\n\n*'Add 100 shares of AAPL at $150'*"

def handle_market_query(mcp_host: MCPHost) -> str:
    """Handle market-related queries"""
    market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
    
    if "error" in market_data:
        return f"❌ I couldn't retrieve market data: {market_data['error']}\n\nPlease check if the market data server is running properly."
    
    # Update session state
    st.session_state.market_data = market_data
    
    indices = market_data.get("indices", {})
    response = "## 📈 Market Overview\n\n"
    
    # Market summary
    positive_count = sum(1 for data in indices.values() if data.get("change_percent", 0) > 0)
    total_count = len(indices)
    
    if positive_count > total_count / 2:
        response += "🟢 **Market Sentiment: Positive** - Most indices are trading higher today.\n\n"
    elif positive_count < total_count / 2:
        response += "🔴 **Market Sentiment: Negative** - Most indices are trading lower today.\n\n"
    else:
        response += "⚪ **Market Sentiment: Mixed** - Markets are showing mixed performance today.\n\n"
    
    # Major indices
    response += "### 📊 Major Indices:\n\n"
    for name, data in indices.items():
        change_emoji = "📈" if data.get("change", 0) > 0 else "📉" if data.get("change", 0) < 0 else "➖"
        price = data.get('current_price', 0)
        change = data.get('change', 0)
        change_percent = data.get('change_percent', 0)
        
        response += f"{change_emoji} **{name}**: {price:.2f} "
        response += f"({change:+.2f}, {change_percent:+.2f}%)\n"
    
    # Add market insights
    response += "\n### 💡 Market Insights:\n\n"
    
    # VIX analysis (if available)
    if "VIX" in indices:
        vix_level = indices["VIX"].get("current_price", 0)
        if vix_level > 30:
            response += "⚠️ **High Volatility**: VIX above 30 indicates high market fear and uncertainty.\n"
        elif vix_level < 15:
            response += "😌 **Low Volatility**: VIX below 15 suggests market complacency and stability.\n"
        else:
            response += "📊 **Moderate Volatility**: VIX levels indicate normal market conditions.\n"
    
    # S&P 500 analysis
    if "S&P 500" in indices:
        sp500_change = indices["S&P 500"].get("change_percent", 0)
        if sp500_change > 1:
            response += "🚀 **Strong Rally**: S&P 500 up over 1% today indicates strong market momentum.\n"
        elif sp500_change < -1:
            response += "📉 **Market Decline**: S&P 500 down over 1% suggests broader market weakness.\n"
    
    return response

def handle_analysis_query(mcp_host: MCPHost, query: str) -> str:
    """Handle stock analysis queries"""
    # Extract stock symbol from query
    words = query.upper().split()
    potential_ticker = None
    
    for word in words:
        if len(word) <= 5 and word.isalpha() and word not in ['ANALYZE', 'ANALYSIS', 'STOCK', 'THE']:
            potential_ticker = word
            break
    
    if not potential_ticker:
        return """
❓ **Stock Symbol Required**

To analyze a stock, please specify the ticker symbol. For example:
- "Analyze AAPL"
- "Give me analysis on GOOGL" 
- "What do you think about TSLA?"

**Popular symbols to try:**
- **AAPL** (Apple)
- **GOOGL** (Google/Alphabet)
- **MSFT** (Microsoft)
- **TSLA** (Tesla)
- **AMZN** (Amazon)
"""
    
    return analyze_stock_comprehensive(mcp_host, potential_ticker)

def analyze_stock_comprehensive(mcp_host: MCPHost, symbol: str) -> str:
    """Comprehensive stock analysis"""
    # Get stock data
    stock_data = mcp_host.call_mcp_tool(
        mcp_host.market_data_server,
        "get_stock_price",
        symbol=symbol,
        period="6mo"
    )
    
    # Get technical analysis
    technical_data = mcp_host.call_mcp_tool(
        mcp_host.market_data_server,
        "get_technical_analysis",
        symbol=symbol
    )
    
    if "error" in stock_data:
        return f"❌ **Analysis Error for {symbol}**\n\n{stock_data['error']}\n\nPlease check if the symbol is correct and try again."
    
    # Build comprehensive response
    company_name = stock_data.get('company_name', symbol)
    current_price = stock_data.get('current_price', 0)
    price_change = stock_data.get('price_change', 0)
    percent_change = stock_data.get('percent_change_pct', 0)
    
    response = f"""
## 📊 {symbol} - {company_name} Analysis

### 💹 Current Trading Data
**Current Price:** ${current_price:.2f}
**Daily Change:** {price_change:+.2f} ({percent_change:+.2f}%)
**Volume:** {stock_data.get('volume', 0):,}

### 📈 Key Fundamentals
**Market Cap:** ${stock_data.get('market_cap', 0):,.0f}
**P/E Ratio:** {stock_data.get('pe_ratio', 0):.2f}
**52W High:** ${stock_data.get('high_52w', 0):.2f}
**52W Low:** ${stock_data.get('low_52w', 0):.2f}
**Beta:** {stock_data.get('beta', 0):.2f}
**Sector:** {stock_data.get('sector', 'N/A')}
"""
    
    # Technical analysis
    if "error" not in technical_data:
        ma = technical_data.get('moving_averages', {})
        rsi = technical_data.get('rsi')
        signals = technical_data.get('signals', [])
        
        response += f"""
### 📊 Technical Indicators
**RSI (14):** {rsi:.2f if rsi else 'N/A'}
**MA20:** ${ma.get('ma_20', 0):.2f if ma.get('ma_20') else 'N/A'}
**MA50:** ${ma.get('ma_50', 0):.2f if ma.get('ma_50') else 'N/A'}
**MA200:**
    ${ma.get('ma_200', 0):.2f if ma.get('ma_200') else 'N/A'}
    """


def refresh_portfolio_data(mcp_host: MCPHost):
    """Refresh portfolio data"""
    with st.spinner("Refreshing portfolio data..."):
        portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
        if "error" not in portfolio_data:
            st.session_state.portfolio_data = portfolio_data
            st.success("Portfolio data refreshed!")
        else:
            st.error(f"Error refreshing portfolio: {portfolio_data['error']}")


def refresh_market_data(mcp_host: MCPHost):
    """Refresh market data"""
    with st.spinner("Refreshing market data..."):
        market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
        if "error" not in market_data:
            st.session_state.market_data = market_data
            st.success("Market data refreshed!")
        else:
            st.error(f"Error refreshing market data: {market_data['error']}")


def refresh_all_data(mcp_host: MCPHost):
    """Refresh all data sources"""
    with st.spinner("Refreshing all data..."):
        # Refresh portfolio
        portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
        if "error" not in portfolio_data:
            st.session_state.portfolio_data = portfolio_data

        # Refresh market data
        market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
        if "error" not in market_data:
            st.session_state.market_data = market_data

        # Update connectivity status
        connectivity = mcp_host.test_server_connectivity()
        st.session_state.connectivity = connectivity
        st.session_state.last_update = datetime.now()

        st.success("All data refreshed!")

main()