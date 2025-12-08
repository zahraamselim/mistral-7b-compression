"""Device specification detection utilities."""

import logging
from typing import Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# GPU specifications database
GPU_SPECS = {
    # NVIDIA Data Center
    't4': {'tdp': 70.0, 'tflops': 8.1, 'memory_gb': 16},
    'p100': {'tdp': 250.0, 'tflops': 18.7, 'memory_gb': 16},
    'v100': {'tdp': 300.0, 'tflops': 28.0, 'memory_gb': 32},
    'a10': {'tdp': 150.0, 'tflops': 31.2, 'memory_gb': 24},
    'a30': {'tdp': 165.0, 'tflops': 41.0, 'memory_gb': 24},
    'a40': {'tdp': 300.0, 'tflops': 37.4, 'memory_gb': 48},
    'a100': {'tdp': 400.0, 'tflops': 78.0, 'memory_gb': 80},
    'h100': {'tdp': 700.0, 'tflops': 204.0, 'memory_gb': 80},
    'h200': {'tdp': 700.0, 'tflops': 204.0, 'memory_gb': 141},
    
    # NVIDIA Gaming/Professional
    'rtx 2060': {'tdp': 160.0, 'tflops': 6.5, 'memory_gb': 6},
    'rtx 2070': {'tdp': 175.0, 'tflops': 7.5, 'memory_gb': 8},
    'rtx 2080': {'tdp': 215.0, 'tflops': 10.1, 'memory_gb': 8},
    'rtx 3060': {'tdp': 170.0, 'tflops': 12.7, 'memory_gb': 12},
    'rtx 3070': {'tdp': 220.0, 'tflops': 20.3, 'memory_gb': 8},
    'rtx 3080': {'tdp': 320.0, 'tflops': 29.8, 'memory_gb': 10},
    'rtx 3090': {'tdp': 350.0, 'tflops': 35.6, 'memory_gb': 24},
    'rtx 4060': {'tdp': 115.0, 'tflops': 15.1, 'memory_gb': 8},
    'rtx 4070': {'tdp': 200.0, 'tflops': 29.1, 'memory_gb': 12},
    'rtx 4080': {'tdp': 320.0, 'tflops': 48.7, 'memory_gb': 16},
    'rtx 4090': {'tdp': 450.0, 'tflops': 82.6, 'memory_gb': 24},
    'rtx 5090': {'tdp': 575.0, 'tflops': 125.0, 'memory_gb': 32},
    'a6000': {'tdp': 300.0, 'tflops': 38.7, 'memory_gb': 48},
    'a5000': {'tdp': 230.0, 'tflops': 27.8, 'memory_gb': 24},
    'rtx 6000': {'tdp': 300.0, 'tflops': 40.0, 'memory_gb': 48},
    
    # AMD
    'mi100': {'tdp': 300.0, 'tflops': 46.1, 'memory_gb': 32},
    'mi210': {'tdp': 300.0, 'tflops': 45.3, 'memory_gb': 64},
    'mi250': {'tdp': 500.0, 'tflops': 90.5, 'memory_gb': 128},
    'mi300': {'tdp': 750.0, 'tflops': 163.0, 'memory_gb': 192},
    
    # Google TPU
    'tpu v3': {'tdp': 450.0, 'tflops': 123.0, 'memory_gb': 16},
    'tpu v4': {'tdp': 450.0, 'tflops': 275.0, 'memory_gb': 32},
}


def get_device_name(is_cuda: bool) -> Optional[str]:
    """Get device name."""
    if not is_cuda or not TORCH_AVAILABLE:
        return None
    
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception as e:
        logger.warning(f"Failed to get device name: {e}")
    
    return None


def get_tdp(is_cuda: bool) -> float:
    """
    Auto-detect TDP based on device.
    
    Args:
        is_cuda: Whether using CUDA
        
    Returns:
        TDP in watts
    """
    if not is_cuda:
        return 15.0  # CPU default
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        logger.warning("CUDA not available, using default TDP")
        return 70.0
    
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        for gpu_key, specs in GPU_SPECS.items():
            if gpu_key in gpu_name:
                logger.info(f"Detected GPU: {gpu_name}")
                logger.info(f"  TDP: {specs['tdp']}W")
                return specs['tdp']
        
        logger.warning(f"Unknown GPU: {gpu_name}, using default TDP: 250W")
        return 250.0
        
    except Exception as e:
        logger.error(f"Failed to detect TDP: {e}")
        return 250.0


def get_peak_tflops(is_cuda: bool) -> float:
    """
    Auto-detect peak TFLOPs based on device.
    
    Args:
        is_cuda: Whether using CUDA
        
    Returns:
        Peak TFLOPs (FP16)
    """
    if not is_cuda:
        return 0.0
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0.0
    
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        for gpu_key, specs in GPU_SPECS.items():
            if gpu_key in gpu_name:
                logger.info(f"  Peak TFLOPs: {specs['tflops']}")
                return specs['tflops']
        
        logger.warning(f"Unknown GPU peak TFLOPs, using default: 8.1")
        return 8.1
        
    except Exception as e:
        logger.error(f"Failed to detect peak TFLOPs: {e}")
        return 8.1


def get_device_specs(is_cuda: bool) -> Dict[str, any]:
    """
    Get comprehensive device specifications.
    
    Args:
        is_cuda: Whether using CUDA
        
    Returns:
        Dictionary with device specs
    """
    specs = {
        'device_type': 'cuda' if is_cuda else 'cpu',
        'device_name': None,
        'tdp_watts': get_tdp(is_cuda),
        'peak_tflops': get_peak_tflops(is_cuda),
        'memory_gb': None,
        'compute_capability': None
    }
    
    if is_cuda and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            specs['device_name'] = torch.cuda.get_device_name(0)
            
            props = torch.cuda.get_device_properties(0)
            specs['memory_gb'] = props.total_memory / (1024**3)
            specs['compute_capability'] = f"{props.major}.{props.minor}"
            
            gpu_name = specs['device_name'].lower()
            for gpu_key, gpu_specs in GPU_SPECS.items():
                if gpu_key in gpu_name:
                    specs['memory_gb'] = gpu_specs['memory_gb']
                    break
            
        except Exception as e:
            logger.warning(f"Failed to get device properties: {e}")
    
    return specs


def print_device_info(is_cuda: bool):
    """Print detailed device information."""
    specs = get_device_specs(is_cuda)
    
    logger.info("="*50)
    logger.info("Device Information")
    logger.info("="*50)
    logger.info(f"Type: {specs['device_type'].upper()}")
    
    if specs['device_name']:
        logger.info(f"Name: {specs['device_name']}")
    
    if specs['memory_gb']:
        logger.info(f"Memory: {specs['memory_gb']:.1f} GB")
    
    if specs['compute_capability']:
        logger.info(f"Compute Capability: {specs['compute_capability']}")
    
    logger.info(f"TDP: {specs['tdp_watts']:.1f} W")
    
    if specs['peak_tflops'] > 0:
        logger.info(f"Peak TFLOPs (FP16): {specs['peak_tflops']:.1f}")
    
    logger.info("="*50)