"""Energy estimation utilities."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def estimate_energy(
    latency_ms_per_token: float,
    tdp_watts: float,
    idle_power_ratio: float = 0.3
) -> float:
    """
    Estimate energy consumption per token.
    
    Energy (J) = Active Power (W) * Time (s)
    Active Power = TDP - Idle Power
    
    Args:
        latency_ms_per_token: Latency in milliseconds per token
        tdp_watts: Thermal Design Power in watts
        idle_power_ratio: Fraction of TDP consumed at idle
        
    Returns:
        Energy consumption in millijoules per token
    """
    if latency_ms_per_token <= 0 or tdp_watts <= 0:
        logger.warning(f"Invalid inputs: latency={latency_ms_per_token}, tdp={tdp_watts}")
        return 0.0
    
    idle_power = tdp_watts * idle_power_ratio
    active_power = tdp_watts - idle_power
    
    latency_seconds = latency_ms_per_token / 1000.0
    
    energy_joules = active_power * latency_seconds
    energy_mj = energy_joules * 1000.0
    
    logger.info(f"Energy per token: {energy_mj:.3f} mJ (active power: {active_power:.1f}W, idle: {idle_power:.1f}W)")
    
    return energy_mj


def estimate_total_energy(
    num_tokens: int,
    energy_per_token_mj: float
) -> Dict[str, float]:
    """
    Estimate total energy consumption.
    
    Args:
        num_tokens: Number of tokens to generate
        energy_per_token_mj: Energy per token in millijoules
        
    Returns:
        Dictionary with energy in different units
    """
    total_mj = num_tokens * energy_per_token_mj
    total_j = total_mj / 1000.0
    total_wh = total_j / 3600.0
    total_kwh = total_wh / 1000.0
    
    return {
        'total_millijoules': total_mj,
        'total_joules': total_j,
        'total_watt_hours': total_wh,
        'total_kilowatt_hours': total_kwh
    }


def estimate_energy_cost(
    num_tokens: int,
    energy_per_token_mj: float,
    electricity_cost_per_kwh: float = 0.12
) -> float:
    """
    Estimate monetary cost of energy consumption.
    
    Args:
        num_tokens: Number of tokens to generate
        energy_per_token_mj: Energy per token in millijoules
        electricity_cost_per_kwh: Cost per kilowatt-hour
        
    Returns:
        Estimated cost in dollars
    """
    energy = estimate_total_energy(num_tokens, energy_per_token_mj)
    cost = energy['total_kilowatt_hours'] * electricity_cost_per_kwh
    
    logger.info(f"Estimated energy cost for {num_tokens:,} tokens: ${cost:.4f}")
    
    return cost


def estimate_carbon_footprint(
    num_tokens: int,
    energy_per_token_mj: float,
    carbon_intensity_g_per_kwh: float = 400.0
) -> float:
    """
    Estimate carbon footprint of inference.
    
    Args:
        num_tokens: Number of tokens to generate
        energy_per_token_mj: Energy per token in millijoules
        carbon_intensity_g_per_kwh: Carbon intensity in grams CO2 per kWh
        
    Returns:
        Carbon emissions in grams of CO2
    """
    energy = estimate_total_energy(num_tokens, energy_per_token_mj)
    carbon_g = energy['total_kilowatt_hours'] * carbon_intensity_g_per_kwh
    
    logger.info(f"Estimated carbon footprint for {num_tokens:,} tokens: {carbon_g:.2f}g CO2")
    
    return carbon_g


def compare_energy_efficiency(
    baseline_energy_mj: float,
    current_energy_mj: float
) -> Dict[str, float]:
    """
    Compare energy efficiency between two models.
    
    Args:
        baseline_energy_mj: Baseline energy per token in mJ
        current_energy_mj: Current energy per token in mJ
        
    Returns:
        Dictionary with comparison metrics
    """
    if baseline_energy_mj <= 0 or current_energy_mj <= 0:
        logger.warning("Invalid energy values for comparison")
        return {
            'energy_reduction_ratio': 0.0,
            'energy_savings_percent': 0.0
        }
    
    reduction_ratio = baseline_energy_mj / current_energy_mj
    savings_percent = ((baseline_energy_mj - current_energy_mj) / baseline_energy_mj) * 100
    
    logger.info(f"Energy efficiency improvement: {reduction_ratio:.2f}x ({savings_percent:.1f}% savings)")
    
    return {
        'energy_reduction_ratio': reduction_ratio,
        'energy_savings_percent': savings_percent
    }