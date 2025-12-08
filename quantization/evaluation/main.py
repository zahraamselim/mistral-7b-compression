"""Main entry point for LLM quantization evaluation."""

import argparse
import logging
import torch
from pathlib import Path

from utils.config import ConfigLoader
from utils.logging import setup_logging
from core.model_interface import create_model_interface
from benchmarks.efficiency import EfficiencyBenchmark
from benchmarks.quality import QualityBenchmark

logger = logging.getLogger(__name__)


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    """Parse torch dtype from string."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float64': torch.float64,
    }
    return dtype_map.get(dtype_str, torch.float16)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate quantized language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Efficiency only:
    python main.py --efficiency

  Quality only:
    python main.py --quality

  Both benchmarks:
    python main.py --efficiency --quality

  Compare with baseline:
    python main.py --efficiency --baseline results/baseline.json

  Custom config:
    python main.py --config my_config.json --efficiency
        """
    )
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file (default: config.json)')
    parser.add_argument('--efficiency', action='store_true',
                        help='Run efficiency benchmarks')
    parser.add_argument('--quality', action='store_true',
                        help='Run quality benchmarks')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./results)')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline results for comparison')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    if not args.efficiency and not args.quality:
        parser.error("Specify at least one benchmark: --efficiency or --quality")
    
    try:
        logger.info("Loading configuration")
        config_loader = ConfigLoader(args.config)
        config = config_loader.get_config()
        
        logger.info("Loading model")
        model_config = config_loader.get_model_config()
        model_interface = create_model_interface(
            model_config.get('interface_type', 'huggingface')
        )
        
        load_kwargs = {
            'model_path': model_config['model_path'],
            'torch_dtype': parse_torch_dtype(model_config.get('torch_dtype', 'float16')),
            'device_map': model_config.get('device_map', 'auto'),
            'trust_remote_code': model_config.get('trust_remote_code', False)
        }
        
        quantization = model_config.get('quantization')
        if quantization:
            load_kwargs.update(quantization)
        
        model_interface.load(**load_kwargs)
        
        output_dir = args.output_dir or config.get('output_dir', './results')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        baseline_results = None
        if args.baseline:
            import json
            with open(args.baseline, 'r') as f:
                baseline_results = json.load(f)
            logger.info(f"Loaded baseline from {args.baseline}")
        
        if args.efficiency:
            logger.info("")
            logger.info("Running efficiency benchmarks")
            
            efficiency_config = config_loader.get_efficiency_config()
            efficiency_benchmark = EfficiencyBenchmark(
                model_interface=model_interface,
                config=efficiency_config,
                verbose=args.verbose
            )
            
            efficiency_results = efficiency_benchmark.run(
                baseline_results=baseline_results
            )
            
            output_path = Path(output_dir) / 'efficiency_results.json'
            efficiency_results.to_json(str(output_path))
            logger.info(f"Results saved to {output_path}")
            
            print(efficiency_results)
        
        if args.quality:
            logger.info("")
            logger.info("Running quality benchmarks")
            
            quality_config = config_loader.get_quality_config()
            quality_benchmark = QualityBenchmark(
                model_interface=model_interface,
                config=quality_config,
                verbose=args.verbose
            )
            
            quality_results = quality_benchmark.run()
            
            output_path = Path(output_dir) / 'quality_results.json'
            quality_results.to_json(str(output_path))
            logger.info(f"Results saved to {output_path}")
            
            print(quality_results)
        
        logger.info("")
        logger.info("Evaluation complete")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())