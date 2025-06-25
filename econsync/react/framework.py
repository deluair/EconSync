"""
ReAct Framework for Economic Reasoning and Acting.
"""

from typing import Dict, List, Any, Optional, Callable
import json
from dataclasses import dataclass
from enum import Enum
import time

from ..core.config import EconSyncConfig
from ..utils.logger import setup_logger


class ActionType(Enum):
    """Types of actions the agent can take."""
    RETRIEVE_DATA = "retrieve_data"
    ANALYZE_TREND = "analyze_trend"
    CALCULATE_STATISTIC = "calculate_statistic"
    SEARCH_LITERATURE = "search_literature"
    FORECAST = "forecast"
    COMPARE_METRICS = "compare_metrics"
    EVALUATE_POLICY = "evaluate_policy"


@dataclass
class Observation:
    """Observation from an action."""
    action_type: ActionType
    result: Any
    success: bool
    timestamp: str
    metadata: Dict[str, Any] = None


@dataclass
class Action:
    """Action to be executed."""
    type: ActionType
    parameters: Dict[str, Any]
    reasoning: str


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_number: int
    thought: str
    action: Optional[Action]
    observation: Optional[Observation]
    confidence: float


class ReActFramework:
    """
    Reasoning and Acting framework for economic analysis.
    Implements the ReAct paradigm: Reason -> Act -> Observe -> Repeat
    """
    
    def __init__(self, config: EconSyncConfig):
        """
        Initialize the ReAct framework.
        
        Args:
            config: EconSync configuration
        """
        self.config = config
        self.logger = setup_logger("ReActFramework", debug=config.debug)
        
        # Available actions
        self.action_registry: Dict[ActionType, Callable] = {}
        self._register_default_actions()
        
        # Reasoning history
        self.reasoning_history: List[ReasoningStep] = []
        
        self.logger.info("ReActFramework initialized successfully")
    
    def _register_default_actions(self):
        """Register default economic analysis actions."""
        self.action_registry = {
            ActionType.RETRIEVE_DATA: self._action_retrieve_data,
            ActionType.ANALYZE_TREND: self._action_analyze_trend,
            ActionType.CALCULATE_STATISTIC: self._action_calculate_statistic,
            ActionType.SEARCH_LITERATURE: self._action_search_literature,
            ActionType.FORECAST: self._action_forecast,
            ActionType.COMPARE_METRICS: self._action_compare_metrics,
            ActionType.EVALUATE_POLICY: self._action_evaluate_policy,
        }
    
    def reason_and_act(self, 
                      query: str, 
                      context: Dict[str, Any],
                      max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Main ReAct loop: reasoning and acting iteratively.
        
        Args:
            query: The economic question or analysis request
            context: Additional context for the analysis
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Complete reasoning trace and final conclusion
        """
        if max_iterations is None:
            max_iterations = self.config.react.max_iterations
        
        self.logger.info(f"Starting ReAct reasoning for query: {query}")
        
        # Initialize reasoning
        self.reasoning_history = []
        
        # Decompose the query into sub-tasks
        initial_thoughts = self._decompose_query(query, context)
        
        for iteration in range(max_iterations):
            self.logger.debug(f"ReAct iteration {iteration + 1}/{max_iterations}")
            
            # Reasoning step
            thought = self._generate_thought(query, context, iteration)
            
            # Decide on action
            action = self._decide_action(thought, query, context)
            
            if action is None:
                # No more actions needed
                break
            
            # Execute action
            observation = self._execute_action(action)
            
            # Update context with observation
            context.update({
                f"observation_{iteration}": observation.result,
                f"action_{iteration}": action.type.value
            })
            
            # Record reasoning step
            step = ReasoningStep(
                step_number=iteration + 1,
                thought=thought,
                action=action,
                observation=observation,
                confidence=self._calculate_confidence(observation)
            )
            self.reasoning_history.append(step)
            
            # Check if we have sufficient information
            if self._should_conclude(query, context, observation):
                break
        
        # Generate final conclusion
        conclusion = self._generate_conclusion(query, context)
        
        result = {
            "query": query,
            "reasoning_steps": [step.__dict__ for step in self.reasoning_history],
            "conclusion": conclusion,
            "total_steps": len(self.reasoning_history),
            "context_used": context
        }
        
        self.logger.info(f"ReAct reasoning completed in {len(self.reasoning_history)} steps")
        return result
    
    def _decompose_query(self, query: str, context: Dict[str, Any]) -> List[str]:
        """
        Decompose complex query into sub-tasks.
        
        Args:
            query: The main query
            context: Available context
            
        Returns:
            List of sub-task descriptions
        """
        # Simple rule-based decomposition (can be enhanced with LLM)
        query_lower = query.lower()
        subtasks = []
        
        if "compare" in query_lower:
            subtasks.append("Identify metrics to compare")
            subtasks.append("Retrieve relevant data")
            subtasks.append("Perform comparison analysis")
        
        if "trend" in query_lower or "forecast" in query_lower:
            subtasks.append("Retrieve historical data")
            subtasks.append("Analyze time series patterns")
            subtasks.append("Generate forecast if needed")
        
        if "policy" in query_lower or "impact" in query_lower:
            subtasks.append("Identify relevant policy measures")
            subtasks.append("Gather policy impact data")
            subtasks.append("Analyze causal relationships")
        
        if "inflation" in query_lower or "gdp" in query_lower or "unemployment" in query_lower:
            subtasks.append("Retrieve macroeconomic indicators")
            subtasks.append("Analyze economic relationships")
        
        # Default decomposition
        if not subtasks:
            subtasks = [
                "Understand the economic question",
                "Identify relevant data sources",
                "Perform quantitative analysis",
                "Synthesize findings"
            ]
        
        return subtasks
    
    def _generate_thought(self, query: str, context: Dict[str, Any], iteration: int) -> str:
        """
        Generate reasoning thought for current iteration.
        
        Args:
            query: Original query
            context: Current context
            iteration: Current iteration number
            
        Returns:
            Reasoning thought
        """
        # Simple template-based thinking (can be enhanced with LLM)
        if iteration == 0:
            return f"I need to analyze the economic question: '{query}'. Let me start by identifying what data and analysis methods are needed."
        
        # Look at previous observations
        recent_observations = [
            context.get(f"observation_{i}") 
            for i in range(max(0, iteration - 2), iteration)
            if context.get(f"observation_{i}") is not None
        ]
        
        if recent_observations:
            return f"Based on the recent observations, I should continue the analysis by examining the relationships and patterns in the data."
        else:
            return f"I need to gather more relevant information to answer the query about {query}."
    
    def _decide_action(self, thought: str, query: str, context: Dict[str, Any]) -> Optional[Action]:
        """
        Decide what action to take based on current thought and context.
        
        Args:
            thought: Current reasoning thought
            query: Original query
            context: Current context
            
        Returns:
            Action to execute or None if no action needed
        """
        query_lower = query.lower()
        
        # Check if we need data
        if not any(f"observation_{i}" in context for i in range(10)):
            if any(keyword in query_lower for keyword in ["gdp", "inflation", "unemployment", "macro"]):
                return Action(
                    type=ActionType.RETRIEVE_DATA,
                    parameters={"dataset": "macroeconomic_indicators"},
                    reasoning="Need macroeconomic data to answer the query"
                )
            elif any(keyword in query_lower for keyword in ["market", "stock", "financial"]):
                return Action(
                    type=ActionType.RETRIEVE_DATA,
                    parameters={"dataset": "financial_markets"},
                    reasoning="Need financial market data for analysis"
                )
            elif any(keyword in query_lower for keyword in ["trade", "export", "import"]):
                return Action(
                    type=ActionType.RETRIEVE_DATA,
                    parameters={"dataset": "trade_data"},
                    reasoning="Need international trade data"
                )
        
        # Check if we need trend analysis
        if "trend" in query_lower and not any("trend_analysis" in str(context.get(f"action_{i}", "")) for i in range(10)):
            return Action(
                type=ActionType.ANALYZE_TREND,
                parameters={"variable": self._extract_variable_from_query(query)},
                reasoning="Need to analyze trends in the specified variable"
            )
        
        # Check if we need forecasting
        if any(keyword in query_lower for keyword in ["forecast", "predict", "future"]):
            return Action(
                type=ActionType.FORECAST,
                parameters={
                    "variable": self._extract_variable_from_query(query),
                    "horizon": 12  # 12 periods ahead
                },
                reasoning="Query requires forecasting future values"
            )
        
        # Check if we need policy evaluation
        if "policy" in query_lower:
            return Action(
                type=ActionType.EVALUATE_POLICY,
                parameters={"policy_description": query},
                reasoning="Query involves policy analysis"
            )
        
        # Default: search literature if no other action taken
        if len([step for step in self.reasoning_history if step.action]) < 2:
            return Action(
                type=ActionType.SEARCH_LITERATURE,
                parameters={"query": query},
                reasoning="Search for relevant economic literature"
            )
        
        return None  # No action needed
    
    def _extract_variable_from_query(self, query: str) -> str:
        """Extract the main economic variable from the query."""
        query_lower = query.lower()
        
        if "gdp" in query_lower:
            return "gdp_growth"
        elif "inflation" in query_lower:
            return "inflation_cpi" 
        elif "unemployment" in query_lower:
            return "unemployment_rate"
        elif "interest" in query_lower or "rate" in query_lower:
            return "federal_funds_rate"
        elif "stock" in query_lower or "market" in query_lower:
            return "sp500_index"
        else:
            return "gdp_growth"  # Default
    
    def _execute_action(self, action: Action) -> Observation:
        """
        Execute the specified action.
        
        Args:
            action: Action to execute
            
        Returns:
            Observation from the action
        """
        self.logger.debug(f"Executing action: {action.type.value}")
        
        try:
            action_func = self.action_registry.get(action.type)
            if action_func is None:
                raise ValueError(f"Unknown action type: {action.type}")
            
            start_time = time.time()
            result = action_func(action.parameters)
            execution_time = time.time() - start_time
            
            observation = Observation(
                action_type=action.type,
                result=result,
                success=True,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            observation = Observation(
                action_type=action.type,
                result=f"Error: {str(e)}",
                success=False,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={"error": str(e)}
            )
        
        return observation
    
    def _action_retrieve_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve economic data."""
        dataset = parameters.get("dataset", "macroeconomic_indicators")
        
        # Simulate data retrieval (in real implementation, would call DataManager)
        return {
            "dataset": dataset,
            "status": "retrieved",
            "shape": (3650, 10),  # 10 years of daily data with 10 variables
            "summary": f"Successfully retrieved {dataset} with economic indicators"
        }
    
    def _action_analyze_trend(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in economic variables."""
        variable = parameters.get("variable", "gdp_growth")
        
        # Simulate trend analysis
        import numpy as np
        
        # Generate sample trend data
        trend_slope = np.random.normal(0.001, 0.0005)
        r_squared = np.random.uniform(0.3, 0.8)
        
        return {
            "variable": variable,
            "trend_slope": trend_slope,
            "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
            "r_squared": r_squared,
            "significance": "significant" if abs(trend_slope) > 0.0005 else "not significant",
            "interpretation": f"The variable {variable} shows a {'positive' if trend_slope > 0 else 'negative'} trend"
        }
    
    def _action_calculate_statistic(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical measures."""
        statistic = parameters.get("statistic", "correlation")
        variables = parameters.get("variables", ["gdp_growth", "inflation_cpi"])
        
        # Simulate statistical calculation
        import numpy as np
        
        if statistic == "correlation":
            correlation = np.random.uniform(-0.5, 0.5)
            return {
                "statistic": statistic,
                "variables": variables,
                "value": correlation,
                "interpretation": f"Correlation between {variables[0]} and {variables[1]} is {correlation:.3f}"
            }
        else:
            return {"statistic": statistic, "result": "Statistic calculated"}
    
    def _action_search_literature(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search economic literature."""
        query = parameters.get("query", "")
        
        # Simulate literature search results
        return {
            "query": query,
            "results_found": 15,
            "top_papers": [
                "Economic Growth and Inflation Dynamics (2023)",
                "Monetary Policy in Modern Economies (2022)",
                "Trade Wars and Global Economic Impact (2024)"
            ],
            "summary": "Found relevant literature on economic relationships and policy impacts"
        }
    
    def _action_forecast(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate economic forecasts."""
        variable = parameters.get("variable", "gdp_growth")
        horizon = parameters.get("horizon", 12)
        
        # Simulate forecasting
        import numpy as np
        
        forecast_values = np.random.normal(0.02, 0.01, horizon)
        confidence_intervals = [(val - 0.01, val + 0.01) for val in forecast_values]
        
        return {
            "variable": variable,
            "horizon": horizon,
            "forecast": forecast_values.tolist(),
            "confidence_intervals": confidence_intervals,
            "method": "ARIMA",
            "accuracy_metrics": {"mae": 0.005, "rmse": 0.008}
        }
    
    def _action_compare_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare economic metrics."""
        metrics = parameters.get("metrics", ["gdp_growth", "inflation_cpi"])
        
        return {
            "metrics": metrics,
            "comparison": "Metrics compared across time periods",
            "key_differences": "Significant variations observed during economic cycles"
        }
    
    def _action_evaluate_policy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy impacts."""
        policy_description = parameters.get("policy_description", "")
        
        return {
            "policy": policy_description,
            "evaluation": "Policy impact analysis completed",
            "expected_effects": "Mixed economic effects anticipated",
            "confidence": 0.7
        }
    
    def _calculate_confidence(self, observation: Observation) -> float:
        """Calculate confidence in the observation."""
        if not observation.success:
            return 0.0
        
        # Simple confidence calculation based on action type
        confidence_map = {
            ActionType.RETRIEVE_DATA: 0.9,
            ActionType.ANALYZE_TREND: 0.8,
            ActionType.CALCULATE_STATISTIC: 0.85,
            ActionType.SEARCH_LITERATURE: 0.7,
            ActionType.FORECAST: 0.6,
            ActionType.COMPARE_METRICS: 0.75,
            ActionType.EVALUATE_POLICY: 0.65
        }
        
        return confidence_map.get(observation.action_type, 0.5)
    
    def _should_conclude(self, query: str, context: Dict[str, Any], observation: Observation) -> bool:
        """Determine if we have enough information to conclude."""
        # Check if we have sufficient observations
        num_observations = len([k for k in context.keys() if k.startswith("observation_")])
        
        if num_observations >= 3:
            return True
        
        # Check if last observation provides conclusive information
        if observation.success and "forecast" in str(observation.result):
            return True
        
        return False
    
    def _generate_conclusion(self, query: str, context: Dict[str, Any]) -> str:
        """Generate final conclusion based on reasoning steps."""
        observations = [
            context.get(f"observation_{i}") 
            for i in range(10) 
            if context.get(f"observation_{i}") is not None
        ]
        
        if not observations:
            return "Unable to reach a conclusion due to insufficient data."
        
        # Simple template-based conclusion
        conclusion = f"Based on the analysis of '{query}', the following insights were gathered:\n\n"
        
        for i, obs in enumerate(observations, 1):
            if isinstance(obs, dict):
                if "trend_direction" in obs:
                    conclusion += f"{i}. Trend analysis shows {obs.get('trend_direction', 'unclear')} patterns.\n"
                elif "forecast" in obs:
                    conclusion += f"{i}. Forecasting indicates future economic conditions.\n"
                elif "dataset" in obs:
                    conclusion += f"{i}. Relevant economic data was successfully retrieved and analyzed.\n"
                else:
                    conclusion += f"{i}. Additional economic analysis was performed.\n"
        
        conclusion += "\nThe economic analysis provides insights into the relationships and patterns relevant to the query."
        
        return conclusion
    
    def analyze_policy_impact(self, 
                            policy_description: str,
                            affected_sectors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Specialized method for policy impact analysis.
        
        Args:
            policy_description: Description of the policy
            affected_sectors: List of affected economic sectors
            
        Returns:
            Policy impact analysis results
        """
        self.logger.info(f"Analyzing policy impact: {policy_description}")
        
        context = {
            "policy_description": policy_description,
            "affected_sectors": affected_sectors or []
        }
        
        query = f"Analyze the economic impact of: {policy_description}"
        
        return self.reason_and_act(query, context) 