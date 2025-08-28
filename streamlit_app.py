"""
Financial AI Assistant - MCP Host UI
A Streamlit interface that acts as an MCP host for the Financial AI System
Save this as: streamlit_app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from typing import Dict, Any, List
import logging
import time

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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-bad { color: #dc3545; }
    .status-warning { color: #ffc107; }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class MCPHost:
    """Simple MCP Host that connects to Financial AI servers"""
    
    def __init__(self):
        # MCP Server URLs - These will be set via environment variables in Docker
        self.portfolio_server = os.getenv("PORTFOLIO_SERVER_URL", "http://ec2-3-90-112-2.compute-1.amazonaws.com:8002")
        self.market_data_server = os.getenv("MARKET_DATA_SERVER_URL", "http://ec2-3-90-112-2.compute-1.amazonaws.com:8001")
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def call_mcp_tool(self, server_url: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the specified MCP server"""
        try:
            # Determine HTTP method based on endpoint
            if tool_name in ["get_portfolio_overview", "get_asset_allocation", "get_portfolio_metrics"]:
                response = requests.get(f"{server_url}/tools/{tool_name}", timeout=30)
            elif tool_name == "get_market_overview":
                response = requests.get(f"{server_url}/tools/{tool_name}", timeout=30)
            else:
                response = requests.post(f"{server_url}/tools/{tool_name}", json=kwargs, timeout=30)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {tool_name}: {e}")
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {e}")
            return {"error": str(e)}
    
    def test_server_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to MCP servers"""
        servers = {
            "Portfolio Server": self.portfolio_server,
            "Market Data Server": self.market_data_server
        }
        
        connectivity = {}
        for name, url in servers.items():
            try:
                response = requests.get(f"{url}/health", timeout=10)
                connectivity[name] = response.status_code == 200
            except:
                connectivity[name] = False
        
        return connectivity

def main():
    """Main Streamlit application"""
    
    # Initialize MCP Host
    mcp_host = MCPHost()
    
    # Main title with custom styling
    st.markdown('<h1 class="main-header">💰 Financial AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-style: italic;">Powered by MCP (Model Context Protocol)</p>', unsafe_allow_html=True)
    
    # Sidebar - Server Status and Navigation
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Test server connectivity with caching
        if st.button("🔄 Refresh Status") or st.session_state.last_update is None or \
           (datetime.now() - st.session_state.last_update).seconds > 60:
            
            with st.spinner("Checking server connectivity..."):
                connectivity = mcp_host.test_server_connectivity()
                st.session_state.connectivity = connectivity
                st.session_state.last_update = datetime.now()
        
        # Display server status
        connectivity = getattr(st.session_state, 'connectivity', {})
        for server, status in connectivity.items():
            if status:
                st.markdown(f'<p class="status-good">✅ {server}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="status-bad">❌ {server}</p>', unsafe_allow_html=True)
        
        st.divider()
        
        # Quick Actions
        st.header("📊 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Portfolio", use_container_width=True):
                get_portfolio_overview(mcp_host)
        
        with col2:
            if st.button("📰 Market", use_container_width=True):
                get_market_overview(mcp_host)
        
        if st.button("🔍 Stock Analysis", use_container_width=True):
            st.session_state.show_stock_analysis = True
        
        if st.button("➕ Add Position", use_container_width=True):
            st.session_state.show_add_position = True
        
        st.divider()
        
        # Server Configuration
        with st.expander("⚙️ Configuration"):
            st.text_input("Portfolio Server", value=mcp_host.portfolio_server, disabled=True)
            st.text_input("Market Data Server", value=mcp_host.market_data_server, disabled=True)
            
            if st.button("Test Connectivity"):
                test_results = mcp_host.test_server_connectivity()
                for server, status in test_results.items():
                    if status:
                        st.success(f"{server}: Connected")
                    else:
                        st.error(f"{server}: Not responding")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat Interface
        st.header("💬 Chat with Financial AI")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and i == len(st.session_state.chat_history) - 1:
                        # Add copy button for latest response
                        if st.button("📋 Copy Response", key=f"copy_{i}"):
                            st.write("Response copied to clipboard!")
        
        # Chat input
        if prompt := st.chat_input("Ask me about your portfolio, stocks, or market analysis..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the query and generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = process_user_query(mcp_host, prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the display
            st.rerun()
    
    with col2:
        # Quick Info Panel
        st.header("📊 Dashboard")
        
        # Portfolio Summary Card
        display_portfolio_summary(mcp_host)
        
        # Market Status Card
        display_market_status(mcp_host)
        
        # Recent Activity
        st.subheader("🕐 Recent Activity")
        if st.session_state.chat_history:
            recent_queries = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"][-3:]
            for query in recent_queries:
                st.text(f"• {query[:50]}...")
        else:
            st.info("No recent activity")
    
    # Additional sections based on user actions
    if hasattr(st.session_state, 'show_stock_analysis') and st.session_state.show_stock_analysis:
        show_stock_analysis_section(mcp_host)
        st.session_state.show_stock_analysis = False
    
    if hasattr(st.session_state, 'show_add_position') and st.session_state.show_add_position:
        show_add_position_section(mcp_host)
        st.session_state.show_add_position = False

def process_user_query(mcp_host: MCPHost, query: str) -> str:
    """Process user query and route to appropriate MCP tools"""
    
    query_lower = query.lower()
    
    try:
        # Portfolio-related queries
        if any(word in query_lower for word in ['portfolio', 'positions', 'holdings', 'my stocks']):
            portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
            
            if "error" in portfolio_data:
                return f"❌ Error getting portfolio data: {portfolio_data['error']}"
            
            st.session_state.portfolio_data = portfolio_data
            
            if portfolio_data.get("status") == "success":
                summary = portfolio_data.get("portfolio_summary", {})
                positions = portfolio_data.get("positions", [])
                
                response = f"""
## 📊 Your Portfolio Overview

**Portfolio Value:** ${summary.get('total_market_value', 0):,.2f}
**Total P&L:** ${summary.get('total_unrealized_pnl', 0):,.2f} ({summary.get('total_return_percent', 0):.2f}%)
**Number of Positions:** {summary.get('number_of_positions', 0)}

### 📈 Top Holdings:
"""
                for pos in positions[:5]:
                    emoji = "📈" if pos['pnl_percent'] > 0 else "📉"
                    response += f"{emoji} **{pos['symbol']}**: {pos['quantity']} shares @ ${pos['current_price']:.2f} (P&L: {pos['pnl_percent']:.2f}%)\n"
                
                if len(positions) > 5:
                    response += f"\n*...and {len(positions) - 5} more positions*"
                
                return response
            else:
                return "❌ No portfolio data available. Add some positions first!"
        
        # Market-related queries
        elif any(word in query_lower for word in ['market', 'indices', 'sp500', 'nasdaq', 'dow']):
            market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
            
            if "error" in market_data:
                return f"❌ Error getting market data: {market_data['error']}"
            
            indices = market_data.get("indices", {})
            response = "## 📈 Market Overview\n\n"
            
            for name, data in indices.items():
                change_emoji = "📈" if data.get("change", 0) > 0 else "📉"
                response += f"{change_emoji} **{name}**: {data.get('current_price', 0):.2f} ({data.get('change_percent', 0):+.2f}%)\n"
            
            return response
        
        # Stock analysis queries
        elif any(word in query_lower for word in ['analyze', 'stock', 'ticker']) and len(query.split()) > 1:
            # Try to extract ticker symbol
            words = query.upper().split()
            potential_ticker = None
            
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    potential_ticker = word
                    break
            
            if potential_ticker:
                return analyze_stock(mcp_host, potential_ticker)
            else:
                return "❓ Please specify a stock ticker symbol (e.g., 'analyze AAPL')"
        
        # Add position queries - enhanced parsing
        elif 'add' in query_lower and any(word in query_lower for word in ['position', 'stock', 'buy', 'share']):
            return parse_add_position_query(mcp_host, query)
        
        # Help/General queries
        else:
            return """
## 🤖 Financial AI Assistant Help

I can help you with:

### 📊 Portfolio Management
- "Show my portfolio" - View your holdings and performance
- "Add 100 shares of AAPL at $150" - Add new positions
- "Portfolio analysis" - Get detailed portfolio insights

### 📈 Market Analysis
- "Market overview" - Current market indices
- "Analyze AAPL" - Technical analysis of specific stocks
- "Market sentiment" - Overall market outlook

### 💡 Investment Insights
- "Risk analysis" - Portfolio risk assessment
- "Recommendations" - Investment suggestions
- "Optimization" - Portfolio improvement advice

### 🚀 Quick Commands
- Type a stock symbol (e.g., "AAPL") for quick analysis
- "Add [number] [symbol] at [price]" to add positions
- "Portfolio" or "positions" to see your holdings
- "Market" for current market status

**Just ask naturally!** I'll understand your questions and route them to the appropriate analysis tools.
"""
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"❌ Error processing your request: {str(e)}"

def parse_add_position_query(mcp_host: MCPHost, query: str) -> str:
    """Parse and process add position queries"""
    try:
        import re
        
        # Extract numbers, symbols, and price from query
        # Patterns like: "add 100 AAPL at 150", "buy 50 shares of GOOGL at $2500"
        
        # Extract quantity
        quantity_match = re.search(r'(\d+)\s*(?:shares?)?', query, re.IGNORECASE)
        quantity = int(quantity_match.group(1)) if quantity_match else None
        
        # Extract symbol
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
        symbol = symbol_match.group(1) if symbol_match else None
        
        # Extract price
        price_match = re.search(r'(?:at|@|\$)\s*(\d+(?:\.\d{2})?)', query, re.IGNORECASE)
        price = float(price_match.group(1)) if price_match else None
        
        if not all([quantity, symbol, price]):
            return """
❓ **Unable to parse position details**

Please use format: "Add [quantity] [symbol] at [price]"

Examples:
- "Add 100 AAPL at 150.00"
- "Buy 50 shares of GOOGL at $2500"
- "Add 25 TSLA at 200.50"

Or use the Add Position form in the sidebar.
"""
        
        # Add the position
        result = mcp_host.call_mcp_tool(
            mcp_host.portfolio_server,
            "add_stock_position",
            symbol=symbol,
            quantity=quantity,
            purchase_price=price
        )
        
        if "error" in result:
            return f"❌ Error adding position: {result['error']}"
        
        if result.get("status") == "success":
            return f"""
✅ **Position Added Successfully!**

- **Symbol:** {result['symbol']}
- **Quantity:** {result['quantity']} shares
- **Purchase Price:** ${result['purchase_price']:.2f}
- **Current Price:** ${result['current_price']:.2f}
- **Market Value:** ${result['market_value']:.2f}
- **Unrealized P&L:** ${result['unrealized_pnl']:.2f}

The position has been {result['action']} to your portfolio.
"""
        else:
            return f"❌ Failed to add position: {result}"
            
    except Exception as e:
        return f"❌ Error processing add position request: {str(e)}"

def analyze_stock(mcp_host: MCPHost, symbol: str) -> str:
    """Analyze a specific stock"""
    try:
        # Get stock price data
        stock_data = mcp_host.call_mcp_tool(
            mcp_host.market_data_server, 
            "get_stock_price", 
            symbol=symbol, 
            period="1mo"
        )
        
        # Get technical analysis
        technical_data = mcp_host.call_mcp_tool(
            mcp_host.market_data_server,
            "get_technical_analysis",
            symbol=symbol
        )
        
        if "error" in stock_data:
            return f"❌ Error analyzing {symbol}: {stock_data['error']}"
        
        # Build response
        change_emoji = "📈" if stock_data.get('percent_change_pct', 0) > 0 else "📉"
        
        response = f"""
## {change_emoji} Stock Analysis: {symbol}

### 📊 Basic Information
- **Current Price:** ${stock_data.get('current_price', 0):.2f}
- **Change:** {stock_data.get('price_change', 0):+.2f} ({stock_data.get('percent_change_pct', 0):+.2f}%)
- **Volume:** {stock_data.get('volume', 0):,}
- **Company:** {stock_data.get('company_name', 'N/A')}
- **Sector:** {stock_data.get('sector', 'N/A')}

### 📈 Key Metrics
- **52W High:** ${stock_data.get('high_52w', 0):.2f}
- **52W Low:** ${stock_data.get('low_52w', 0):.2f}
- **P/E Ratio:** {stock_data.get('pe_ratio', 0):.2f}
- **Market Cap:** ${stock_data.get('market_cap', 0):,.0f}
- **Beta:** {stock_data.get('beta', 0):.2f}
"""
        
        if "error" not in technical_data:
            ma = technical_data.get('moving_averages', {})
            rsi = technical_data.get('rsi')
            signals = technical_data.get('signals', [])
            
            response += f"""
### 📊 Technical Analysis
- **RSI:** {rsi:.2f if rsi else 'N/A'} {'(Overbought)' if rsi and rsi > 70 else '(Oversold)' if rsi and rsi < 30 else ''}
- **MA20:** ${ma.get('ma_20', 0):.2f if ma.get('ma_20') else 'N/A'}
- **MA50:** ${ma.get('ma_50', 0):.2f if ma.get('ma_50') else 'N/A'}
- **MA200:** ${ma.get('ma_200', 0):.2f if ma.get('ma_200') else 'N/A'}

### 🎯 Trading Signals
"""
            if signals:
                for signal in signals:
                    signal_emoji = "🟢" if "bullish" in signal.lower() or "buy" in signal.lower() else "🔴" if "bearish" in signal.lower() or "sell" in signal.lower() else "⚠️"
                    response += f"{signal_emoji} {signal}\n"
            else:
                response += "No specific signals at this time"
        
        return response
    
    except Exception as e:
        return f"❌ Error analyzing {symbol}: {str(e)}"

def get_portfolio_overview(mcp_host: MCPHost):
    """Get and display portfolio overview"""
    with st.spinner("Loading portfolio data..."):
        portfolio_data = mcp_host.call_mcp_tool(mcp_host.portfolio_server, "get_portfolio_overview")
        st.session_state.portfolio_data = portfolio_data
        
        if "error" in portfolio_data:
            st.error(f"Error: {portfolio_data['error']}")
        else:
            st.success("Portfolio data loaded successfully!")
            st.rerun()

def get_market_overview(mcp_host: MCPHost):
    """Get and display market overview"""
    with st.spinner("Loading market data..."):
        market_data = mcp_host.call_mcp_tool(mcp_host.market_data_server, "get_market_overview")
        
        if "error" in market_data:
            st.error(f"Error: {market_data['error']}")
        else:
            st.success("Market data loaded!")
            # Create and show a simple chart
            indices = market_