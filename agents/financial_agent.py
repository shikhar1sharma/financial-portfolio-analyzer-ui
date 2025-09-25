"""
True Agentic Financial Agent
Implements ReAct pattern with dynamic planning and execution
"""

import openai
import json
from typing import Dict, Any, List, Optional, Tuple
import logging
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

class AgenticTool:
    """Represents a tool the agent can use"""
    def __init__(self, name: str, description: str, server_url: str, endpoint: str, method: str = "POST", parameters: Dict = None):
        self.name = name
        self.description = description
        self.server_url = server_url
        self.endpoint = endpoint
        self.method = method
        self.parameters = parameters or {}

class AgenticMemory:
    """Agent memory system"""

    def __init__(self):
        self.working_memory = []  # Current conversation context
        self.tool_usage_history = []  # Track successful tool combinations
        self.user_preferences = {}  # Learn user patterns

    def add_to_working_memory(self, thought: str, action: str, observation: str):
        """Add reasoning step to working memory"""
        self.working_memory.append({
            "thought": thought,
            "action": action,
            "observation": observation,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_summary(self) -> str:
        """Summarize current context for agent"""
        if not self.working_memory:
            return "No previous context."

        summary = "Previous steps in this analysis:\n"
        for i, step in enumerate(self.working_memory[-5:], 1):  # Last 5 steps
            summary += f"{i}. Thought: {step['thought'][:100]}...\n"
            summary += f"   Action: {step['action']}\n"
            summary += f"   Found: {step['observation'][:100]}...\n\n"
        return summary

    def record_successful_strategy(self, goal: str, strategy: List[str], outcome: str):
        """Learn from successful tool usage patterns"""
        self.tool_usage_history.append({
            "goal": goal,
            "strategy": strategy,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })

class TrueAgenticFinancialAgent:
    """
    True agentic financial agent that plans and executes dynamically
    Uses ReAct pattern: Reasoning + Acting
    """

    def __init__(self, model: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)

        # Setup OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.model = model
        self.client = openai.OpenAI()

        # TODO: Change based on env
        # Server endpoints
        self.portfolio_server = os.getenv("PORTFOLIO_SERVER_URL", "http://localhost:8002")
        self.market_data_server = os.getenv("MARKET_DATA_SERVER_URL", "http://localhost:8001")

        # Initialize memory and tools
        self.memory = AgenticMemory()
        self.tools = self._initialize_tools()

        # Agent configuration
        self.max_iterations = 15
        self.confidence_threshold = 0.8

    # TODO: add new tools here
    def _initialize_tools(self) -> Dict[str, AgenticTool]:
        """Initialize available tools from both servers"""
        tools = {
            # Portfolio Server Tools
            "get_portfolio_overview": AgenticTool(
                name="get_portfolio_overview",
                description="Get complete portfolio summary including positions, P&L, allocation, and metrics",
                server_url=self.portfolio_server,
                endpoint="/tools/get_portfolio_overview",
                method="GET"
            ),

            "add_stock_position": AgenticTool(
                name="add_stock_position",
                description="Add a new stock position to portfolio",
                server_url=self.portfolio_server,
                endpoint="/tools/add_stock_position",
                method="POST",
                parameters={"symbol": "str", "quantity": "float", "purchase_price": "float"}
            ),

            "calculate_returns": AgenticTool(
                name="calculate_returns",
                description="Calculate portfolio returns for specific timeframe (1D, 1W, 1M, 3M, 6M, 1Y)",
                server_url=self.portfolio_server,
                endpoint="/tools/calculate_returns",
                method="POST",
                parameters={"timeframe": "str"}
            ),

            "get_asset_allocation": AgenticTool(
                name="get_asset_allocation",
                description="Get portfolio asset allocation breakdown by sector and position",
                server_url=self.portfolio_server,
                endpoint="/tools/get_asset_allocation",
                method="GET"
            ),

            "get_portfolio_metrics": AgenticTool(
                name="get_portfolio_metrics",
                description="Get advanced portfolio metrics like Sharpe ratio, beta, volatility, max drawdown",
                server_url=self.portfolio_server,
                endpoint="/tools/get_portfolio_metrics",
                method="GET"
            ),

            # Market Data Server Tools
            "get_stock_price": AgenticTool(
                name="get_stock_price",
                description="Get current stock price, company info, and basic financial metrics",
                server_url=self.market_data_server,
                endpoint="/tools/get_stock_price",
                method="POST",
                parameters={"symbol": "str", "period": "str"}
            ),

            "get_market_overview": AgenticTool(
                name="get_market_overview",
                description="Get major market indices performance and overall market sentiment",
                server_url=self.market_data_server,
                endpoint="/tools/get_market_overview",
                method="GET"
            ),

            "get_technical_analysis": AgenticTool(
                name="get_technical_analysis",
                description="Get technical indicators (RSI, MACD, moving averages, Bollinger bands) and trading signals",
                server_url=self.market_data_server,
                endpoint="/tools/get_technical_analysis",
                method="POST",
                parameters={"symbol": "str"}
            ),

            "get_portfolio_performance": AgenticTool(
                name="get_portfolio_performance",
                description="Calculate performance metrics for a weighted portfolio of stocks",
                server_url=self.market_data_server,
                endpoint="/tools/get_portfolio_performance",
                method="POST",
                parameters={"symbols": "List[str]", "weights": "List[float]"}
            )
        }

        return tools

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool with error handling and logging"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        tool = self.tools[tool_name]

        try:
            if tool.method == "GET":
                response = requests.get(f"{tool.server_url}{tool.endpoint}", timeout=30)
            else:
                response = requests.post(
                    f"{tool.server_url}{tool.endpoint}",
                    json=kwargs,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

            response.raise_for_status()
            result = response.json()

            self.logger.info(f"Tool {tool_name} executed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}

    def _generate_reasoning(self, user_goal: str, iteration: int) -> str:
        """Generate agent's reasoning about what to do next"""

        context = self.memory.get_context_summary()
        available_tools = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

        reasoning_prompt = f"""
        You are an expert financial advisor AI agent. You help users achieve their financial goals through systematic analysis.

        USER GOAL: "{user_goal}"
        
        CURRENT CONTEXT:
        {context}
        
        AVAILABLE TOOLS:
        {available_tools}
        
        ITERATION: {iteration}/{self.max_iterations}
        
        CRITICAL: You must gather actual data before providing recommendations. Do not give generic advice without first using tools to understand the user's specific situation.
        
        Your task: Think step-by-step about what you should do next to help achieve the user's goal.
        
        MANDATORY STEPS FOR PORTFOLIO QUESTIONS:
        1. ALWAYS start with get_portfolio_overview to understand current positions
        2. If user has positions, analyze each major holding individually  
        3. Check market conditions for context
        4. Only then provide specific recommendations
        
        CONFIDENCE RULES:
        - Start with CONFIDENCE: 0.1 if you haven't called any tools yet
        - Only reach CONFIDENCE: 0.8+ after gathering specific user data
        - Generic advice without user-specific data should have CONFIDENCE: 0.2 max
        
        Respond with your reasoning in this format:
        THOUGHT: [Your reasoning about what to do next and why - be specific about data needs]
        CONFIDENCE: [How confident are you? Start low, build up as you gather data]
        NEXT_ACTION: [Specific tool to call next, or "SYNTHESIZE" if you have enough user-specific data]
        TOOL_PARAMETERS: [JSON object with parameters for the tool, or null if synthesizing]
        
        Remember: Gather data first, then analyze, then recommend.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return f"THOUGHT: Error in reasoning process: {e}\nCONFIDENCE: 0.0\nNEXT_ACTION: SYNTHESIZE\nTOOL_PARAMETERS: null"

    def _parse_reasoning_response(self, reasoning_text: str) -> Tuple[str, float, str, Dict]:
        """Parse the agent's reasoning response"""
        try:
            thought_match = re.search(r'THOUGHT:\s*(.*?)(?=\nCONFIDENCE:|$)', reasoning_text, re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', reasoning_text)
            action_match = re.search(r'NEXT_ACTION:\s*(.*?)(?=\n|$)', reasoning_text)
            params_match = re.search(r'TOOL_PARAMETERS:\s*(.*?)(?=\n|$)', reasoning_text, re.DOTALL)

            thought = thought_match.group(1).strip() if thought_match else "No reasoning provided"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            action = action_match.group(1).strip() if action_match else "SYNTHESIZE"

            # Parse parameters
            params_text = params_match.group(1).strip() if params_match else "null"
            if params_text.lower() == "null" or not params_text:
                parameters = {}
            else:
                try:
                    parameters = json.loads(params_text)
                except json.JSONDecodeError:
                    parameters = {}

            return thought, confidence, action, parameters

        except Exception as e:
            self.logger.error(f"Error parsing reasoning: {e}")
            return "Error parsing reasoning", 0.0, "SYNTHESIZE", {}

    def _synthesize_final_answer(self, user_goal: str) -> str:
        """Generate final comprehensive answer based on gathered information"""

        context = self.memory.get_context_summary()

        synthesis_prompt = f"""
        You are an expert financial advisor providing final recommendations.

        USER'S ORIGINAL GOAL: "{user_goal}"
        
        ANALYSIS PERFORMED:
        {context}
        
        Based on all the information you've gathered and analyzed, provide a comprehensive response that:
        
        1. **EXECUTIVE SUMMARY**: Directly answer the user's question with key findings
        2. **KEY INSIGHTS**: Most important discoveries from your analysis  
        3. **SPECIFIC RECOMMENDATIONS**: Actionable steps prioritized by importance
        4. **RISK ASSESSMENT**: Any risks or concerns identified
        5. **NEXT STEPS**: What the user should do immediately vs. longer-term
        
        Format your response professionally with clear sections and bullet points.
        Make recommendations specific and actionable (e.g., "Reduce GOOGL position by 30%" not "Consider reducing tech exposure").
        
        Provide a confidence level for your recommendations and explain any assumptions made.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error synthesizing answer: {e}")
            return f"I encountered an error while synthesizing my analysis: {e}"

    def analyze(self, user_goal: str) -> Dict[str, Any]:
        """
        Main agentic analysis method - this is what gets called from UI
        Implements ReAct pattern: Reasoning + Acting in a loop
        """

        self.logger.info(f"Starting agentic analysis for goal: {user_goal}")

        # Reset memory for new analysis
        self.memory.working_memory = []

        analysis_log = []
        final_confidence = 0.0

        for iteration in range(1, self.max_iterations + 1):

            # REASONING PHASE: Agent thinks about what to do next
            reasoning_text = self._generate_reasoning(user_goal, iteration)
            thought, confidence, next_action, parameters = self._parse_reasoning_response(reasoning_text)

            analysis_log.append(f"**Iteration {iteration}:**")
            analysis_log.append(f"🧠 **Thinking:** {thought}")

            # Check if agent is confident enough to synthesize
            if confidence >= self.confidence_threshold or next_action == "SYNTHESIZE":
                analysis_log.append(
                    f"✅ **Decision:** Ready to synthesize recommendations (confidence: {confidence:.1%})")
                final_confidence = confidence
                break

            # ACTION PHASE: Agent executes chosen tool
            if next_action in self.tools:
                analysis_log.append(f"🔧 **Action:** Using {next_action}")

                tool_result = self.call_tool(next_action, **parameters)

                # OBSERVATION PHASE: Agent observes and learns from results
                if "error" in tool_result:
                    observation = f"Tool failed: {tool_result['error']}"
                    analysis_log.append(f"❌ **Result:** {observation}")
                else:
                    # Summarize key findings from tool result
                    observation = self._summarize_tool_result(next_action, tool_result)
                    analysis_log.append(f"📊 **Found:** {observation}")

                # Add to memory
                self.memory.add_to_working_memory(thought, next_action, observation)

            else:
                analysis_log.append(f"❌ **Error:** Unknown action '{next_action}'")
                break

        # SYNTHESIS PHASE: Generate final recommendations
        analysis_log.append("🎯 **Synthesizing final recommendations...**")
        final_answer = self._synthesize_final_answer(user_goal)

        return {
            "status": "success",
            "user_goal": user_goal,
            "final_answer": final_answer,
            "confidence": final_confidence,
            "iterations_used": len(self.memory.working_memory),
            "analysis_log": "\n".join(analysis_log),
            "timestamp": datetime.now().isoformat()
        }

    def _summarize_tool_result(self, tool_name: str, result: Dict) -> str:
        """Summarize key findings from tool results for agent memory"""
        try:
            if tool_name == "get_portfolio_overview":
                if result.get("status") == "success":
                    summary = result.get("portfolio_summary", {})
                    positions = result.get("positions", [])
                    return f"Portfolio: ${summary.get('total_market_value', 0):,.0f} value, {len(positions)} positions, {summary.get('total_return_percent', 0):+.1f}% return"
                else:
                    return "No portfolio positions found"

            elif tool_name == "get_market_overview":
                indices = result.get("indices", {})
                if indices:
                    sp500 = indices.get("S&P 500", {})
                    nasdaq = indices.get("NASDAQ", {})
                    return f"Market: S&P 500 {sp500.get('change_percent', 0):+.1f}%, NASDAQ {nasdaq.get('change_percent', 0):+.1f}%"
                else:
                    return "Market data retrieved"

            elif tool_name == "get_technical_analysis":
                symbol = result.get("symbol", "Unknown")
                rsi = result.get("rsi")
                signals = result.get("signals", [])
                signal_summary = ", ".join(signals[:2]) if signals else "No clear signals"
                return f"{symbol}: RSI {rsi:.0f}, {signal_summary}"

            elif tool_name == "get_stock_price":
                symbol = result.get("symbol", "Unknown")
                price = result.get("current_price", 0)
                change_pct = result.get("percent_change_pct", 0)
                return f"{symbol}: ${price:.2f} ({change_pct:+.1f}%)"

            else:
                return "Data retrieved successfully"

        except Exception as e:
            return f"Data retrieved with parsing issues: {e}"

    # Convenience methods for common use cases
    def portfolio_health_check(self) -> Dict[str, Any]:
        """Convenience method for portfolio health analysis"""
        return self.analyze("Analyze my portfolio health and provide specific recommendations for improvement")

    def stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """Convenience method for individual stock analysis"""
        return self.analyze(f"Provide comprehensive analysis and recommendation for {symbol} stock")

    def market_outlook(self) -> Dict[str, Any]:
        """Convenience method for market analysis"""
        return self.analyze("What is the current market outlook and how should it affect my investment strategy?")

    def optimization_advice(self) -> Dict[str, Any]:
        """Convenience method for portfolio optimization"""
        return self.analyze("How can I optimize my portfolio allocation for better risk-adjusted returns?")

# Utility functions for testing and validation
def test_agent_connectivity() -> Dict[str, Any]:
    """Test agent setup and tool connectivity"""
    agent = TrueAgenticFinancialAgent()

    connectivity = {}
    for tool_name, tool in agent.tools.items():
        try:
            # Simple connectivity test
            if tool.method == "GET":
                response = requests.get(f"{tool.server_url}/health", timeout=5)
            else:
                response = requests.get(f"{tool.server_url}/health", timeout=5)

            connectivity[tool_name] = response.status_code == 200
        except:
            connectivity[tool_name] = False

    return {
        "timestamp": datetime.now().isoformat(),
        "tool_connectivity": connectivity,
        "all_tools_ready": all(connectivity.values()),
        "available_tools": list(agent.tools.keys())
    }