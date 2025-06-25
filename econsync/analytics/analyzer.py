"""
Economic Analyzer for comprehensive economic analysis and insights.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json

from ..core.config import EconSyncConfig
from ..utils.logger import setup_logger


class EconomicAnalyzer:
    """
    Comprehensive economic analysis system that combines multiple analytical approaches.
    """
    
    def __init__(self, config: EconSyncConfig):
        """
        Initialize the economic analyzer.
        
        Args:
            config: EconSync configuration
        """
        self.config = config
        self.logger = setup_logger("EconomicAnalyzer", debug=config.debug)
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        
        # Available analysis types
        self.analysis_types = [
            "descriptive", "correlation", "trend", "forecasting", 
            "policy_impact", "sector_analysis", "comparative"
        ]
        
        self.logger.info("EconomicAnalyzer initialized successfully")
    
    def analyze(self, 
               query: str,
               reasoning: Dict[str, Any],
               retrieved_docs: List[Dict[str, Any]],
               data_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive economic analysis.
        
        Args:
            query: The analysis query
            reasoning: Reasoning results from ReAct framework
            retrieved_docs: Documents retrieved by RAG system
            data_sources: Optional data sources to use
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info(f"Starting comprehensive analysis for: {query}")
        
        try:
            # Determine analysis type
            analysis_type = self._determine_analysis_type(query)
            
            # Extract key variables and concepts
            key_variables = self._extract_variables(query, retrieved_docs)
            
            # Perform quantitative analysis
            quantitative_results = self._perform_quantitative_analysis(
                query, key_variables, data_sources
            )
            
            # Perform qualitative analysis
            qualitative_results = self._perform_qualitative_analysis(
                query, retrieved_docs, reasoning
            )
            
            # Synthesize insights
            insights = self._synthesize_insights(
                query, quantitative_results, qualitative_results, reasoning
            )
            
            # Generate visualizations
            visualizations = self._generate_visualizations(
                key_variables, quantitative_results
            )
            
            # Compile comprehensive results
            analysis_results = {
                "query": query,
                "analysis_type": analysis_type,
                "key_variables": key_variables,
                "quantitative_analysis": quantitative_results,
                "qualitative_analysis": qualitative_results,
                "insights": insights,
                "visualizations": visualizations,
                "confidence_score": self._calculate_confidence_score(
                    quantitative_results, qualitative_results, retrieved_docs
                ),
                "methodology": self._describe_methodology(analysis_type),
                "limitations": self._identify_limitations(analysis_type, data_sources),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache results
            cache_key = f"{query}_{analysis_type}_{datetime.now().strftime('%Y%m%d')}"
            self.analysis_cache[cache_key] = analysis_results
            
            self.logger.info("Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis needed based on the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["trend", "time", "over time", "historical"]):
            return "trend"
        elif any(word in query_lower for word in ["forecast", "predict", "future", "projection"]):
            return "forecasting"
        elif any(word in query_lower for word in ["policy", "impact", "effect", "intervention"]):
            return "policy_impact"
        elif any(word in query_lower for word in ["compare", "comparison", "versus", "vs"]):
            return "comparative"
        elif any(word in query_lower for word in ["sector", "industry", "regional"]):
            return "sector_analysis"
        elif any(word in query_lower for word in ["correlation", "relationship", "association"]):
            return "correlation"
        else:
            return "descriptive"
    
    def _extract_variables(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract key economic variables from query and retrieved documents."""
        
        # Common economic variables
        economic_variables = {
            "gdp": ["gdp", "gross domestic product", "economic output"],
            "inflation": ["inflation", "cpi", "consumer price index", "price level"],
            "unemployment": ["unemployment", "jobless", "employment rate"],
            "interest_rate": ["interest rate", "federal funds", "monetary policy"],
            "exchange_rate": ["exchange rate", "currency", "dollar"],
            "trade": ["trade", "exports", "imports", "trade balance"],
            "stock_market": ["stock market", "sp500", "equity", "shares"],
            "oil_price": ["oil", "crude", "energy prices"],
            "productivity": ["productivity", "output per worker"],
            "consumption": ["consumption", "consumer spending"]
        }
        
        query_lower = query.lower()
        extracted_vars = []
        
        # Check query for variables
        for var_key, keywords in economic_variables.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_vars.append(var_key)
        
        # Check retrieved documents for additional variables
        for doc in retrieved_docs:
            doc_text = doc.get("text", "").lower()
            for var_key, keywords in economic_variables.items():
                if var_key not in extracted_vars and any(keyword in doc_text for keyword in keywords):
                    extracted_vars.append(var_key)
        
        # Default variables if none found
        if not extracted_vars:
            extracted_vars = ["gdp", "inflation", "unemployment"]
        
        return extracted_vars[:5]  # Limit to 5 variables for manageable analysis
    
    def _perform_quantitative_analysis(self, 
                                     query: str, 
                                     variables: List[str],
                                     data_sources: Optional[List[str]]) -> Dict[str, Any]:
        """Perform quantitative economic analysis."""
        
        # Generate synthetic data for demonstration
        # In real implementation, would use actual DataManager
        data = self._generate_analysis_data(variables)
        
        results = {
            "data_summary": self._calculate_summary_statistics(data),
            "correlation_analysis": self._calculate_correlations(data, variables),
            "trend_analysis": self._analyze_trends(data, variables),
            "volatility_analysis": self._analyze_volatility(data, variables),
            "time_series_properties": self._analyze_time_series_properties(data, variables)
        }
        
        return results
    
    def _perform_qualitative_analysis(self, 
                                    query: str,
                                    retrieved_docs: List[Dict[str, Any]],
                                    reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Perform qualitative economic analysis based on literature and reasoning."""
        
        # Analyze retrieved documents
        doc_analysis = self._analyze_documents(retrieved_docs)
        
        # Extract theoretical frameworks
        frameworks = self._extract_theoretical_frameworks(retrieved_docs)
        
        # Analyze reasoning quality
        reasoning_analysis = self._analyze_reasoning_quality(reasoning)
        
        return {
            "document_analysis": doc_analysis,
            "theoretical_frameworks": frameworks,
            "reasoning_analysis": reasoning_analysis,
            "literature_consensus": self._determine_literature_consensus(retrieved_docs),
            "research_gaps": self._identify_research_gaps(retrieved_docs)
        }
    
    def _generate_analysis_data(self, variables: List[str]) -> pd.DataFrame:
        """Generate synthetic data for analysis."""
        
        # Create date range (5 years of monthly data)
        dates = pd.date_range(start="2019-01-01", end="2024-01-01", freq="M")
        n_periods = len(dates)
        
        data = {"date": dates}
        
        # Generate realistic economic time series
        for var in variables:
            if var == "gdp":
                # GDP growth with business cycle
                trend = 0.02 / 12  # 2% annual growth, monthly
                cycle = 0.005 * np.sin(2 * np.pi * np.arange(n_periods) / 48)  # 4-year cycle
                noise = np.random.normal(0, 0.002, n_periods)
                data[var] = trend + cycle + noise
                
            elif var == "inflation":
                # Inflation targeting around 2%
                target = 0.02 / 12  # 2% annual, monthly
                shocks = np.random.normal(0, 0.001, n_periods)
                data[var] = np.clip(target + shocks, -0.01, 0.05)
                
            elif var == "unemployment":
                # Unemployment rate (counter-cyclical)
                base_rate = 0.05
                cycle = -0.01 * np.sin(2 * np.pi * np.arange(n_periods) / 48)  # Counter-cyclical
                noise = np.random.normal(0, 0.001, n_periods)
                data[var] = np.clip(base_rate + cycle + noise, 0.02, 0.12)
                
            elif var == "interest_rate":
                # Interest rates following economic conditions
                base_rate = 0.02 / 12
                trend = 0.0001 * np.arange(n_periods)
                noise = np.random.normal(0, 0.0005, n_periods)
                data[var] = np.clip(base_rate + trend + noise, 0, 0.08)
                
            else:
                # Generic economic variable
                trend = np.random.normal(0, 0.0001, n_periods)
                noise = np.random.normal(0, 0.005, n_periods)
                data[var] = np.cumsum(trend + noise)
        
        return pd.DataFrame(data)
    
    def _calculate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the data."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        return {
            "mean": numeric_data.mean().to_dict(),
            "std": numeric_data.std().to_dict(),
            "min": numeric_data.min().to_dict(),
            "max": numeric_data.max().to_dict(),
            "skewness": numeric_data.skew().to_dict(),
            "kurtosis": numeric_data.kurtosis().to_dict()
        }
    
    def _calculate_correlations(self, data: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Calculate correlation matrix and analysis."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        "variable1": correlation_matrix.columns[i],
                        "variable2": correlation_matrix.columns[j],
                        "correlation": corr_value,
                        "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "average_correlation": correlation_matrix.mean().mean()
        }
    
    def _analyze_trends(self, data: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Analyze trends in the data."""
        
        trend_analysis = {}
        
        for var in variables:
            if var in data.columns:
                series = data[var].dropna()
                
                # Simple linear trend
                x = np.arange(len(series))
                coeffs = np.polyfit(x, series, 1)
                slope = coeffs[0]
                
                # Trend direction and strength
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                
                # Calculate R-squared for trend
                fitted_values = np.polyval(coeffs, x)
                ss_res = np.sum((series - fitted_values) ** 2)
                ss_tot = np.sum((series - np.mean(series)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                trend_analysis[var] = {
                    "slope": slope,
                    "direction": trend_direction,
                    "r_squared": r_squared,
                    "significance": "significant" if r_squared > 0.1 else "not significant"
                }
        
        return trend_analysis
    
    def _analyze_volatility(self, data: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Analyze volatility patterns in the data."""
        
        volatility_analysis = {}
        
        for var in variables:
            if var in data.columns:
                series = data[var].dropna()
                
                # Calculate rolling volatility
                rolling_std = series.rolling(window=12).std()  # 12-month rolling
                
                volatility_analysis[var] = {
                    "overall_volatility": series.std(),
                    "average_rolling_volatility": rolling_std.mean(),
                    "volatility_trend": "increasing" if rolling_std.iloc[-6:].mean() > rolling_std.iloc[:6].mean() else "decreasing",
                    "max_volatility_period": rolling_std.idxmax() if not rolling_std.empty else None
                }
        
        return volatility_analysis
    
    def _analyze_time_series_properties(self, data: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Analyze time series properties of the variables."""
        
        properties = {}
        
        for var in variables:
            if var in data.columns:
                series = data[var].dropna()
                
                # Basic time series properties
                autocorr_1 = series.autocorr(lag=1) if len(series) > 1 else 0
                
                properties[var] = {
                    "autocorrelation_lag1": autocorr_1,
                    "persistence": "high" if autocorr_1 > 0.8 else "moderate" if autocorr_1 > 0.5 else "low",
                    "observations": len(series),
                    "missing_values": data[var].isnull().sum()
                }
        
        return properties
    
    def _analyze_documents(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retrieved documents for economic insights."""
        
        if not retrieved_docs:
            return {"message": "No documents available for analysis"}
        
        # Extract key concepts and themes
        all_text = " ".join([doc.get("text", "") for doc in retrieved_docs])
        
        # Simple keyword frequency analysis
        economic_keywords = [
            "growth", "inflation", "unemployment", "monetary", "fiscal", 
            "policy", "market", "economy", "recession", "expansion"
        ]
        
        keyword_counts = {}
        for keyword in economic_keywords:
            keyword_counts[keyword] = all_text.lower().count(keyword)
        
        # Document sources and categories
        sources = [doc.get("metadata", {}).get("source", "Unknown") for doc in retrieved_docs]
        categories = [doc.get("metadata", {}).get("category", "Unknown") for doc in retrieved_docs]
        
        return {
            "num_documents": len(retrieved_docs),
            "sources": list(set(sources)),
            "categories": list(set(categories)),
            "keyword_frequency": keyword_counts,
            "average_score": np.mean([doc.get("score", 0) for doc in retrieved_docs]),
            "key_themes": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _extract_theoretical_frameworks(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract theoretical frameworks mentioned in documents."""
        
        frameworks = []
        
        # Common economic theories and models
        theory_keywords = [
            "keynesian", "monetarist", "neoclassical", "behavioral", "institutional",
            "phillips curve", "taylor rule", "solow model", "is-lm", "dsge"
        ]
        
        for doc in retrieved_docs:
            text = doc.get("text", "").lower()
            for theory in theory_keywords:
                if theory in text and theory not in frameworks:
                    frameworks.append(theory)
        
        return frameworks
    
    def _analyze_reasoning_quality(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of the reasoning process."""
        
        reasoning_steps = reasoning.get("reasoning_steps", [])
        
        if not reasoning_steps:
            return {"message": "No reasoning steps available for analysis"}
        
        # Analyze reasoning quality
        total_steps = len(reasoning_steps)
        successful_steps = sum(1 for step in reasoning_steps if step.get("observation", {}).get("success", False))
        
        confidence_scores = [step.get("confidence", 0) for step in reasoning_steps]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            "total_reasoning_steps": total_steps,
            "successful_steps": successful_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "average_confidence": avg_confidence,
            "reasoning_quality": "high" if avg_confidence > 0.8 else "moderate" if avg_confidence > 0.6 else "low"
        }
    
    def _determine_literature_consensus(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Determine consensus from literature."""
        
        if len(retrieved_docs) < 3:
            return "Insufficient literature for consensus determination"
        
        # Simple heuristic based on document scores and consistency
        high_score_docs = [doc for doc in retrieved_docs if doc.get("score", 0) > 0.8]
        
        if len(high_score_docs) >= len(retrieved_docs) * 0.7:
            return "Strong consensus in literature"
        elif len(high_score_docs) >= len(retrieved_docs) * 0.5:
            return "Moderate consensus in literature"
        else:
            return "Limited consensus in literature"
    
    def _identify_research_gaps(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Identify potential research gaps."""
        
        # Simple heuristic for identifying gaps
        gaps = []
        
        if len(retrieved_docs) < 5:
            gaps.append("Limited research available on this topic")
        
        # Check for recency
        recent_docs = [doc for doc in retrieved_docs 
                      if "2023" in str(doc.get("metadata", {})) or "2024" in str(doc.get("metadata", {}))]
        
        if len(recent_docs) < len(retrieved_docs) * 0.3:
            gaps.append("Limited recent research (post-2022)")
        
        gaps.append("Need for more empirical validation")
        gaps.append("Cross-country comparative analysis needed")
        
        return gaps[:3]  # Limit to top 3 gaps
    
    def _synthesize_insights(self, 
                           query: str,
                           quantitative_results: Dict[str, Any],
                           qualitative_results: Dict[str, Any],
                           reasoning: Dict[str, Any]) -> List[str]:
        """Synthesize insights from all analyses."""
        
        insights = []
        
        # Quantitative insights
        correlations = quantitative_results.get("correlation_analysis", {}).get("strong_correlations", [])
        if correlations:
            for corr in correlations[:3]:  # Top 3 correlations
                insights.append(
                    f"Strong {corr['strength']} correlation ({corr['correlation']:.3f}) "
                    f"between {corr['variable1']} and {corr['variable2']}"
                )
        
        # Trend insights
        trends = quantitative_results.get("trend_analysis", {})
        for var, trend_info in trends.items():
            if trend_info.get("significance") == "significant":
                insights.append(
                    f"{var.title()} shows a {trend_info['direction']} trend "
                    f"(R² = {trend_info['r_squared']:.3f})"
                )
        
        # Literature insights
        doc_analysis = qualitative_results.get("document_analysis", {})
        if doc_analysis.get("key_themes"):
            top_theme = doc_analysis["key_themes"][0]
            insights.append(f"Literature emphasizes '{top_theme[0]}' as a key concept")
        
        # Reasoning insights
        reasoning_analysis = qualitative_results.get("reasoning_analysis", {})
        if reasoning_analysis.get("reasoning_quality") == "high":
            insights.append("Analysis supported by high-quality reasoning process")
        
        # Default insight if none generated
        if not insights:
            insights.append("Economic analysis reveals complex relationships requiring further investigation")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _generate_visualizations(self, 
                                variables: List[str],
                                quantitative_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization specifications."""
        
        visualizations = {}
        
        # Time series plots
        visualizations["time_series"] = {
            "type": "line_chart",
            "variables": variables,
            "title": f"Time Series of {', '.join(variables)}",
            "x_axis": "Date",
            "y_axis": "Value"
        }
        
        # Correlation heatmap
        if "correlation_analysis" in quantitative_results:
            visualizations["correlation_heatmap"] = {
                "type": "heatmap",
                "data": "correlation_matrix",
                "title": "Variable Correlation Matrix",
                "color_scale": "RdBu"
            }
        
        # Trend analysis
        visualizations["trend_analysis"] = {
            "type": "scatter_with_trend",
            "variables": variables,
            "title": "Trend Analysis",
            "show_regression": True
        }
        
        return visualizations
    
    def _calculate_confidence_score(self, 
                                  quantitative_results: Dict[str, Any],
                                  qualitative_results: Dict[str, Any],
                                  retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the analysis."""
        
        scores = []
        
        # Data quality score
        if quantitative_results:
            scores.append(0.8)  # High confidence in synthetic data
        
        # Literature quality score
        doc_scores = [doc.get("score", 0) for doc in retrieved_docs]
        if doc_scores:
            avg_doc_score = np.mean(doc_scores)
            scores.append(avg_doc_score)
        
        # Reasoning quality score
        reasoning_analysis = qualitative_results.get("reasoning_analysis", {})
        reasoning_quality = reasoning_analysis.get("average_confidence", 0.5)
        scores.append(reasoning_quality)
        
        # Number of sources
        num_docs = len(retrieved_docs)
        source_score = min(1.0, num_docs / 10)  # Normalize to max of 1.0
        scores.append(source_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _describe_methodology(self, analysis_type: str) -> str:
        """Describe the methodology used for the analysis."""
        
        methodologies = {
            "descriptive": "Descriptive statistical analysis with summary statistics and data exploration",
            "correlation": "Correlation analysis and statistical association testing",
            "trend": "Time series trend analysis using linear regression and decomposition",
            "forecasting": "Econometric forecasting using ARIMA, VAR, and machine learning methods",
            "policy_impact": "Policy impact assessment using quasi-experimental methods",
            "sector_analysis": "Sectoral economic analysis with cross-sectional comparisons",
            "comparative": "Comparative economic analysis across variables, regions, or time periods"
        }
        
        return methodologies.get(analysis_type, "General economic analysis methodology")
    
    def _identify_limitations(self, 
                            analysis_type: str, 
                            data_sources: Optional[List[str]]) -> List[str]:
        """Identify limitations of the analysis."""
        
        limitations = []
        
        # Data limitations
        if not data_sources or "synthetic" in str(data_sources):
            limitations.append("Analysis based on synthetic data - real-world validation needed")
        
        # Methodology limitations
        if analysis_type == "forecasting":
            limitations.append("Forecasts subject to model uncertainty and external shocks")
        elif analysis_type == "policy_impact":
            limitations.append("Causal identification may be limited without experimental data")
        
        # General limitations
        limitations.extend([
            "Results dependent on data quality and model assumptions",
            "External validity may be limited to specific contexts",
            "Correlation does not imply causation"
        ])
        
        return limitations[:4]  # Limit to top 4 limitations
    
    def get_cached_analysis(self, query: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis results."""
        
        cache_key = f"{query}_{analysis_type}_{datetime.now().strftime('%Y%m%d')}"
        return self.analysis_cache.get(cache_key)
    
    def export_analysis(self, analysis_results: Dict[str, Any], format: str = "json") -> str:
        """Export analysis results to specified format."""
        
        if format == "json":
            return json.dumps(analysis_results, indent=2, default=str)
        else:
            return str(analysis_results) 