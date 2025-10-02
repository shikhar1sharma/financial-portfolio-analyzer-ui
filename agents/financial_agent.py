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
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.performance_stats = {
            'total_tool_calls': 0,
            'parallel_batches': 0,
            'sequential_calls': 0,
            'total_tool_time': 0.0,
            'average_tool_time': 0.0,
            'fastest_call': float('inf'),
            'slowest_call': 0.0,
            'tool_timings': {}  # Track timing per tool type
        }

    def add_to_working_memory(self, thought: str, action: str, observation: str, execution_time: float = 0.0):
        """Add reasoning step to working memory"""
        self.working_memory.append({
            "thought": thought,
            "action": action,
            "observation": observation,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        })

        # Update performance stats
        if execution_time > 0:
            self.performance_stats['total_tool_calls'] += 1
            self.performance_stats['total_tool_time'] += execution_time
            self.performance_stats['average_tool_time'] = (
                    self.performance_stats['total_tool_time'] / self.performance_stats['total_tool_calls']
            )
            self.performance_stats['fastest_call'] = min(
                self.performance_stats['fastest_call'], execution_time
            )
            self.performance_stats['slowest_call'] = max(
                self.performance_stats['slowest_call'], execution_time
            )

            # Track per-tool timing
            if action not in self.performance_stats['tool_timings']:
                self.performance_stats['tool_timings'][action] = []
            self.performance_stats['tool_timings'][action].append(execution_time)

    def get_performance_report(self) -> str:
        """Generate performance report"""
        stats = self.performance_stats

        report = f"""
            📊 PERFORMANCE STATISTICS:
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            • Total Tool Calls: {stats['total_tool_calls']}
            • Parallel Batches: {stats['parallel_batches']}
            • Sequential Calls: {stats['sequential_calls']}
            • Total Time in Tools: {stats['total_tool_time']:.2f}s
            • Average Call Time: {stats['average_tool_time']:.2f}s
            • Fastest Call: {stats['fastest_call']:.2f}s
            • Slowest Call: {stats['slowest_call']:.2f}s
            
            PER-TOOL BREAKDOWN:
            """

        for tool_name, timings in stats['tool_timings'].items():
            avg_time = sum(timings) / len(timings)
            report += f"  • {tool_name}: {len(timings)} calls, avg {avg_time:.2f}s\n"

        return report

    def get_context_summary(self) -> str:
        """Summarize current context for agent"""
        if not self.working_memory:
            return "No previous context."

        summary = "Previous steps in this analysis:\n"
        for i, step in enumerate(self.working_memory[-5:], 1):  # Last 5 steps
            exec_time = step.get('execution_time', 0)
            time_str = f" ({exec_time:.2f}s)" if exec_time > 0 else ""
            summary += f"{i}. Thought: {step['thought'][:100]}...\n"
            summary += f"   Action: {step['action']}{time_str}\n"
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
        self.reasoning_model = "gpt-4o-mini" # Fast for routing
        self.synthesis_model = "gpt-4"       # Smart for final answer

        # Setup OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.model = model
        self.client = openai.OpenAI(api_key=openai.api_key)

        # TODO: Change based on env
        # Server endpoints
        self.portfolio_server = os.getenv("PORTFOLIO_SERVER_URL", "http://localhost:8002")
        self.market_data_server = os.getenv("MARKET_DATA_SERVER_URL", "http://localhost:8001")

        # Initialize memory and tools
        self.memory = AgenticMemory()
        self.tools = self._initialize_tools()

        # Agent configuration
        self.max_iterations = 15
        self.confidence_threshold = 0.5

        # Parallel execution setup
        self.max_parallel_tools = 5
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_tools)

        # Performance tracking
        self.iteration_timings = []
        self.llm_timings = []

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
        start_time = time.time()

        try:
            self.logger.info(f"🔧 Calling tool: {tool_name} with params: {kwargs}")

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

            execution_time = time.time() - start_time

            self.logger.info(f"✅ Tool {tool_name} completed in {execution_time:.2f}s")
            return result, execution_time

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}

    def _execute_tools_parallel(self, tool_names: List[str], parameters_dict: Dict[str, Dict]) -> Tuple[
        Dict[str, Any], float]:
        """Execute multiple tools in parallel with comprehensive timing"""

        if not tool_names:
            return {}, 0.0

        batch_start_time = time.time()
        self.logger.info(f"🚀 Starting parallel execution of {len(tool_names)} tools: {', '.join(tool_names)}")

        def call_single_tool(tool_name: str) -> Tuple[str, Dict[str, Any], float]:
            """Execute single tool and return result with timing"""
            params = parameters_dict.get(tool_name, {})
            result, exec_time = self.call_tool(tool_name, **params)
            return tool_name, result, exec_time

        # Submit all tools to thread pool
        futures = {
            self.executor.submit(call_single_tool, tool_name): tool_name
            for tool_name in tool_names
        }

        # Collect results as they complete
        results = {}
        tool_timings = {}

        for future in as_completed(futures):
            try:
                tool_name, result, exec_time = future.result(timeout=30)
                results[tool_name] = result
                tool_timings[tool_name] = exec_time
                self.logger.info(f"  ✓ {tool_name} finished in {exec_time:.2f}s")
            except Exception as e:
                tool_name = futures[future]
                results[tool_name] = {"error": str(e)}
                tool_timings[tool_name] = 0.0
                self.logger.error(f"  ✗ {tool_name} failed: {e}")

        total_parallel_time = time.time() - batch_start_time

        # Log parallel execution summary
        sequential_time = sum(tool_timings.values())
        speedup = sequential_time / total_parallel_time if total_parallel_time > 0 else 1.0

        self.logger.info(
            f"🎯 Parallel batch completed:\n"
            f"   • Wall time: {total_parallel_time:.2f}s\n"
            f"   • Sequential time would be: {sequential_time:.2f}s\n"
            f"   • Speedup: {speedup:.2f}x\n"
            f"   • Efficiency: {(speedup / len(tool_names)) * 100:.1f}%"
        )

        # Update memory stats
        self.memory.performance_stats['parallel_batches'] += 1

        return results, total_parallel_time

    def _generate_reasoning_batch(self, user_goal: str, iteration: int) -> Tuple[str, float]:
        """Generate agent's reasoning with parallel support and timing"""

        context = self.memory.get_context_summary()
        available_tools = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

        reasoning_prompt = f"""
        You are an expert financial advisor AI agent.

        USER GOAL: "{user_goal}"

        CURRENT CONTEXT:
        {context}

        AVAILABLE TOOLS:
        {available_tools}

        ITERATION: {iteration}/{self.max_iterations}

        CRITICAL EFFICIENCY RULES:
        1. You can suggest MULTIPLE INDEPENDENT TOOLS to run in parallel
        2. For portfolio questions, simultaneously fetch:
           - Portfolio overview
           - Market overview  
           - Individual stock data for major holdings
        3. Only sequential execution when tool B requires output from tool A
        4. MINIMIZE iterations by batching independent data gathering

        CONFIDENCE RULES:
        - Start with CONFIDENCE: 0.1 if no tools called yet
        - Reach CONFIDENCE: 0.8+ after gathering comprehensive data
        - Plan to gather ALL needed data in 2-3 parallel batches

        Respond with:
        THOUGHT: [Your reasoning about what data you need]
        CONFIDENCE: [0.0-1.0]
        PARALLEL_ACTIONS: [JSON list of tools to run simultaneously]
        SEQUENTIAL_ACTION: [Single tool needing previous results, or "SYNTHESIZE"]
        TOOL_PARAMETERS: [JSON object mapping tool names to parameters]

        Example:
        PARALLEL_ACTIONS: ["get_portfolio_overview", "get_market_overview"]
        TOOL_PARAMETERS: {{"get_portfolio_overview": {{}}, "get_market_overview": {{}}}}
        """

        try:
            llm_start = time.time()
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            llm_time = time.time() - llm_start

            self.llm_timings.append(('reasoning', llm_time))
            self.logger.info(f"🧠 LLM reasoning completed in {llm_time:.2f}s")

            return response.choices[0].message.content, llm_time

        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return (
                f"THOUGHT: Error\n"
                f"CONFIDENCE: 0.0\n"
                f"PARALLEL_ACTIONS: []\n"
                f"SEQUENTIAL_ACTION: SYNTHESIZE\n"
                f"TOOL_PARAMETERS: {{}}",
                0.0
            )

    def _parse_reasoning_batch_response(self, reasoning_text: str) -> Tuple[str, float, List[str], str, Dict]:
        """Parse reasoning that includes parallel actions"""
        try:
            thought_match = re.search(r'THOUGHT:\s*(.*?)(?=\nCONFIDENCE:|$)', reasoning_text, re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', reasoning_text)
            parallel_match = re.search(r'PARALLEL_ACTIONS:\s*(\[.*?\])', reasoning_text, re.DOTALL)
            sequential_match = re.search(r'SEQUENTIAL_ACTION:\s*(.*?)(?=\n|$)', reasoning_text)
            params_match = re.search(r'TOOL_PARAMETERS:\s*(\{.*?\})', reasoning_text, re.DOTALL)

            thought = thought_match.group(1).strip() if thought_match else "No reasoning"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5

            # Parse parallel actions
            parallel_text = parallel_match.group(1) if parallel_match else "[]"
            try:
                parallel_actions = json.loads(parallel_text)
            except:
                parallel_actions = []

            sequential_action = sequential_match.group(1).strip() if sequential_match else "SYNTHESIZE"

            # Parse parameters
            params_text = params_match.group(1) if params_match else "{}"
            try:
                all_parameters = json.loads(params_text)
            except:
                all_parameters = {}

            return thought, confidence, parallel_actions, sequential_action, all_parameters

        except Exception as e:
            self.logger.error(f"Error parsing reasoning: {e}")
            return "Error parsing", 0.0, [], "SYNTHESIZE", {}

    def _synthesize_final_answer(self, user_goal: str) -> Tuple[str, float]:
        """Generate final answer with timing"""

        context = self.memory.get_context_summary()

        synthesis_prompt = f"""
        You are an expert financial advisor providing final recommendations.

        USER'S GOAL: "{user_goal}"

        ANALYSIS PERFORMED:
        {context}

        Provide comprehensive response with:
        1. **EXECUTIVE SUMMARY**: Key findings
        2. **KEY INSIGHTS**: Important discoveries
        3. **SPECIFIC RECOMMENDATIONS**: Actionable steps
        4. **RISK ASSESSMENT**: Concerns identified
        5. **NEXT STEPS**: Immediate vs longer-term actions
        """

        try:
            llm_start = time.time()
            response = self.client.chat.completions.create(
                model=self.synthesis_model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            llm_time = time.time() - llm_start

            self.llm_timings.append(('synthesis', llm_time))
            self.logger.info(f"🎯 LLM synthesis completed in {llm_time:.2f}s")

            return response.choices[0].message.content, llm_time

        except Exception as e:
            self.logger.error(f"Error synthesizing: {e}")
            return f"Error: {e}", 0.0

    def analyze(self, user_goal: str) -> Dict[str, Any]:
        """
        Main analysis with parallel execution and timing
        """

        analysis_start_time = time.time()
        self.logger.info(f"🚀 Starting analysis: {user_goal}")

        # Reset tracking
        self.memory.working_memory = []
        self.memory.performance_stats = {
            'total_tool_calls': 0,
            'parallel_batches': 0,
            'sequential_calls': 0,
            'total_tool_time': 0.0,
            'average_tool_time': 0.0,
            'fastest_call': float('inf'),
            'slowest_call': 0.0,
            'tool_timings': {}
        }
        self.iteration_timings = []
        self.llm_timings = []

        analysis_log = []
        final_confidence = 0.0

        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.time()

            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"📍 ITERATION {iteration}/{self.max_iterations}")
            self.logger.info(f"{'=' * 60}")

            # REASONING PHASE
            reasoning_text, reasoning_time = self._generate_reasoning_batch(user_goal, iteration)
            thought, confidence, parallel_actions, sequential_action, all_params = \
                self._parse_reasoning_batch_response(reasoning_text)

            analysis_log.append(f"\n**Iteration {iteration}:** ({reasoning_time:.2f}s reasoning)")
            analysis_log.append(f"🧠 **Thinking:** {thought}")
            analysis_log.append(f"📊 **Confidence:** {confidence:.1%}")

            # Check if ready
            if confidence >= self.confidence_threshold or (
                    sequential_action == "SYNTHESIZE" and not parallel_actions
            ):
                analysis_log.append(f"✅ **Ready to synthesize** (confidence: {confidence:.1%})")
                final_confidence = confidence
                break

            # PARALLEL EXECUTION
            if parallel_actions:
                self.logger.info(f"🚀 Executing {len(parallel_actions)} tools in parallel...")
                analysis_log.append(
                    f"🚀 **Parallel Batch:** {len(parallel_actions)} tools → {', '.join(parallel_actions)}"
                )

                parallel_results, parallel_time = self._execute_tools_parallel(
                    parallel_actions, all_params
                )

                analysis_log.append(f"⏱️  **Parallel execution:** {parallel_time:.2f}s")

                # Process results
                individual_times = []
                for tool_name, result in parallel_results.items():
                    if "error" in result:
                        observation = f"failed: {result['error']}"
                        analysis_log.append(f"   ❌ {tool_name}: {observation}")
                    else:
                        observation = self._summarize_tool_result(tool_name, result)
                        analysis_log.append(f"   ✅ {tool_name}: {observation}")

                    tool_times = self.memory.performance_stats['tool_timings'].get(tool_name, [])
                    if tool_times:
                        individual_times.append(tool_times[-1])
                        self.memory.add_to_working_memory(
                            f"Parallel: {tool_name}",
                            tool_name,
                            observation,
                            tool_times[-1]
                        )

                # Show speedup
                if individual_times:
                    sequential_would_be = sum(individual_times)
                    speedup = sequential_would_be / parallel_time if parallel_time > 0 else 1.0
                    analysis_log.append(
                        f"   ⚡ **Speedup:** {speedup:.2f}x (sequential: {sequential_would_be:.2f}s)"
                    )

            # SEQUENTIAL EXECUTION
            if sequential_action != "SYNTHESIZE" and sequential_action in self.tools:
                self.logger.info(f"🔧 Sequential: {sequential_action}")
                analysis_log.append(f"🔧 **Sequential:** {sequential_action}")

                params = all_params.get(sequential_action, {})
                result, exec_time = self.call_tool(sequential_action, **params)

                self.memory.performance_stats['sequential_calls'] += 1

                if "error" in result:
                    observation = f"failed: {result['error']}"
                    analysis_log.append(f"   ❌ ({exec_time:.2f}s): {observation}")
                else:
                    observation = self._summarize_tool_result(sequential_action, result)
                    analysis_log.append(f"   ✅ ({exec_time:.2f}s): {observation}")

                self.memory.add_to_working_memory(thought, sequential_action, observation, exec_time)

            iteration_time = time.time() - iteration_start
            self.iteration_timings.append(iteration_time)

            analysis_log.append(f"⏱️  **Iteration total:** {iteration_time:.2f}s")
            self.logger.info(f"✓ Iteration {iteration} completed in {iteration_time:.2f}s\n")

        # SYNTHESIS
        self.logger.info(f"🎯 Synthesizing...")
        analysis_log.append("\n🎯 **Synthesizing final recommendations...**")

        final_answer, synthesis_time = self._synthesize_final_answer(user_goal)
        analysis_log.append(f"⏱️  **Synthesis:** {synthesis_time:.2f}s")

        # Performance summary
        total_time = time.time() - analysis_start_time
        total_llm_time = sum(t for _, t in self.llm_timings)
        total_tool_time = self.memory.performance_stats['total_tool_time']

        perf_summary = self._generate_performance_summary(
            total_time, total_llm_time, total_tool_time
        )

        self.logger.info(f"\n{perf_summary}")
        analysis_log.append(f"\n{perf_summary}")

        return {
            "status": "success",
            "user_goal": user_goal,
            "final_answer": final_answer,
            "confidence": final_confidence,
            "iterations_used": len(self.iteration_timings),
            "analysis_log": "\n".join(analysis_log),
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": {
                "total_time": total_time,
                "total_llm_time": total_llm_time,
                "total_tool_time": total_tool_time,
                "parallel_batches": self.memory.performance_stats['parallel_batches'],
                "sequential_calls": self.memory.performance_stats['sequential_calls'],
                "speedup_achieved": self._calculate_speedup()
            }
        }

    def _generate_performance_summary(self, total: float, llm: float, tools: float) -> str:
        """Generate performance summary"""

        stats = self.memory.performance_stats
        llm_pct = (llm / total * 100) if total > 0 else 0
        tool_pct = (tools / total * 100) if total > 0 else 0

        summary = f"""
                    ╔════════════════════════════════════════════════════════╗
                    ║              PERFORMANCE ANALYSIS                       ║
                    ╠════════════════════════════════════════════════════════╣
                    ║ TIMING:                                                 ║
                    ║   • Total:          {total:>8.2f}s  (100.0%)            ║
                    ║   • LLM Time:       {llm:>8.2f}s  ({llm_pct:>5.1f}%)    ║
                    ║   • Tool Time:      {tools:>8.2f}s  ({tool_pct:>5.1f}%) ║
                    ╠════════════════════════════════════════════════════════╣
                    ║ TOOLS:                                                  ║
                    ║   • Total Calls:    {stats['total_tool_calls']:>8}      ║
                    ║   • Parallel Batches: {stats['parallel_batches']:>6}    ║
                    ║   • Sequential:     {stats['sequential_calls']:>8}      ║
                    ║   • Avg Time:       {stats['average_tool_time']:>8.2f}s ║
                    ║   • Speedup:        {self._calculate_speedup():>8.2f}x  ║
                    ╚════════════════════════════════════════════════════════╝
                    """

        if stats['tool_timings']:
            summary += "\nPER-TOOL:\n"
            for tool, timings in sorted(stats['tool_timings'].items()):
                avg = sum(timings) / len(timings)
                summary += f"  • {tool:<25} {len(timings):>2} calls, {avg:>5.2f}s avg\n"

        return summary

    def _calculate_speedup(self) -> float:
        """Calculate speedup from parallelization"""
        stats = self.memory.performance_stats
        if stats['parallel_batches'] == 0:
            return 1.0

        # Rough estimate based on tool calls
        parallel_calls = stats['total_tool_calls'] - stats['sequential_calls']
        if parallel_calls > 0 and stats['parallel_batches'] > 0:
            return parallel_calls / stats['parallel_batches']
        return 1.0

    def _summarize_tool_result(self, tool_name: str, result: Dict) -> str:
        """Summarize tool results"""
        try:
            if tool_name == "get_portfolio_overview":
                if result.get("status") == "success":
                    summary = result.get("portfolio_summary", {})
                    positions = result.get("positions", [])
                    return f"Portfolio: ${summary.get('total_market_value', 0):,.0f}, {len(positions)} positions, {summary.get('total_return_percent', 0):+.1f}%"
                return "No portfolio"

            elif tool_name == "get_market_overview":
                indices = result.get("indices", {})
                if indices:
                    sp500 = indices.get("S&P 500", {})
                    return f"S&P 500: {sp500.get('change_percent', 0):+.1f}%"
                return "Market data retrieved"

            elif tool_name == "get_technical_analysis":
                symbol = result.get("symbol", "?")
                rsi = result.get("rsi")
                return f"{symbol}: RSI {rsi:.0f}" if rsi else f"{symbol} analyzed"

            elif tool_name == "get_stock_price":
                symbol = result.get("symbol", "?")
                price = result.get("current_price", 0)
                change = result.get("percent_change_pct", 0)
                return f"{symbol}: ${price:.2f} ({change:+.1f}%)"

            else:
                return "Data retrieved"

        except Exception as e:
            return f"Data retrieved (parse error: {e})"

    # Convenience methods
    def portfolio_health_check(self) -> Dict[str, Any]:
        """Convenience: portfolio health analysis"""
        return self.analyze("Analyze my portfolio health and provide specific recommendations")

    def stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """Convenience: stock analysis"""
        return self.analyze(f"Provide comprehensive analysis for {symbol} stock")

    def market_outlook(self) -> Dict[str, Any]:
        """Convenience: market analysis"""
        return self.analyze("What is the current market outlook?")

    def optimization_advice(self) -> Dict[str, Any]:
        """Convenience: portfolio optimization"""
        return self.analyze("How can I optimize my portfolio allocation?")

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Utility functions for testing and validation
def test_agent_connectivity() -> Dict[str, Any]:
    """Test agent setup and tool connectivity"""
    agent = TrueAgenticFinancialAgent()

    connectivity = {}
    for tool_name, tool in agent.tools.items():
        try:
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