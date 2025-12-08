# LLM Quantization Evaluation Suite

Comprehensive evaluation framework for quantized large language models.

## Features

- **Efficiency Benchmarks**: Latency, throughput, memory, energy, FLOPs, MFU
- **Quality Benchmarks**: Perplexity, task accuracy via lm-eval-harness
- **Model Support**: HuggingFace, GPTQ, AWQ, HQQ
- **Analysis Tools**: Comparison, visualization, reporting, export

## Installation

```bash
pip install -r requirements.txt
```

### Core Dependencies

```bash
pip install torch transformers datasets evaluate
```

### Optional Dependencies

```bash
# For quantized models
pip install auto-gptq  # GPTQ
pip install autoawq    # AWQ
pip install hqq        # HQQ

# For task evaluation
pip install lm-eval

# For visualization
pip install matplotlib seaborn

# For statistical analysis
pip install scipy
```

## Quick Start

### 1. Configure Your Model

Edit `config.json`:

```json
{
  "model": {
    "model_path": "meta-llama/Llama-2-7b-chat-hf",
    "interface_type": "huggingface",
    "torch_dtype": "float16"
  }
}
```

### 2. Run Benchmarks

```bash
# Efficiency only
python main.py --efficiency

# Quality only
python main.py --quality

# Both benchmarks
python main.py --efficiency --quality

# With baseline comparison
python main.py --efficiency --baseline results/baseline.json
```

### 3. Analyze Results

```bash
# Compare results
python -m analysis.comparator results/model1.json results/model2.json

# Generate visualizations
python -m analysis.visualizer results/*.json --output-dir plots/

# Create HTML report
python -m analysis.reporter results/*.json --output report.html

# Export to CSV/Markdown/LaTeX
python -m analysis.exporter results/*.json --output results.csv --format csv
```

## Project Structure

```
llm-quant-eval/
├── core/                       # Core abstractions
│   ├── model_interface.py      # Model interface
│   ├── benchmark.py            # Base benchmark class
│   └── result.py               # Result dataclass
│
├── models/                     # Model implementations
│   ├── huggingface.py          # Standard HF models
│   ├── gptq.py                 # GPTQ quantized
│   ├── awq.py                  # AWQ quantized
│   └── hqq.py                  # HQQ quantized
│
├── benchmarks/                 # Evaluation benchmarks
│   ├── efficiency/             # Efficiency benchmarks
│   │   ├── benchmark.py        # Main orchestrator
│   │   ├── latency.py          # Latency measurement
│   │   ├── throughput.py       # Throughput measurement
│   │   ├── memory.py           # Memory profiling
│   │   ├── compute.py          # FLOPs/MFU calculation
│   │   ├── energy.py           # Energy estimation
│   │   └── device.py           # Device detection
│   │
│   └── quality/                # Quality benchmarks
│       ├── benchmark.py        # Main orchestrator
│       ├── perplexity.py       # Perplexity calculation
│       └── tasks.py            # lm-eval wrapper
│
├── analysis/                   # Analysis tools
│   ├── comparator.py           # Compare results
│   ├── visualizer.py           # Generate plots
│   ├── reporter.py             # HTML reports
│   └── exporter.py             # Export utilities
│
├── utils/                      # Utilities
│   ├── config.py               # Config loading
│   └── logging.py              # Logging setup
│
├── main.py                     # CLI entry point
├── config.json                 # Configuration
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Configuration

### Model Configuration

```json
{
  "model": {
    "model_path": "model-name-or-path",
    "interface_type": "huggingface", // or "gptq", "awq", "hqq"
    "torch_dtype": "float16",
    "device_map": "auto",
    "quantization": {
      // Quantization-specific options
    }
  }
}
```

### Efficiency Benchmarks

```json
{
  "benchmarks": {
    "efficiency": {
      "num_warmup": 3,
      "num_runs": 10,
      "max_new_tokens": 128,
      "prompts": ["prompt1", "prompt2"],
      "measure_prefill_decode": true,
      "measure_batch_throughput": false
    }
  }
}
```

### Quality Benchmarks

```json
{
  "benchmarks": {
    "quality": {
      "perplexity": {
        "enabled": true,
        "dataset": "wikitext",
        "num_samples": 100
      },
      "tasks": {
        "enabled": false,
        "task_list": {
          "hellaswag": {
            "enabled": false,
            "num_fewshot": 0
          }
        }
      }
    }
  }
}
```

## Usage Examples

### Evaluate Quantized Model

```python
from core.model_interface import create_model_interface
from benchmarks.efficiency import EfficiencyBenchmark
from benchmarks.quality import QualityBenchmark

# Load quantized model
model = create_model_interface('gptq')
model.load('TheBloke/Llama-2-7B-GPTQ')

# Run efficiency benchmark
efficiency = EfficiencyBenchmark(model, config)
eff_results = efficiency.run()

# Run quality benchmark
quality = QualityBenchmark(model, config)
qual_results = quality.run()
```

### Compare Models

```python
from analysis.comparator import ResultsComparator

comparator = ResultsComparator()
comparator.load_results([
    'results/fp16.json',
    'results/int4.json'
], names=['FP16', 'INT4'])

# Print comparison
comparator.print_comparison(0, 1)

# Statistical significance
comparator.print_significance_test(0, 1, metrics=['latency_ms_per_token'])
```

### Generate Visualizations

```python
from analysis.visualizer import ResultsVisualizer

viz = ResultsVisualizer()
viz.load_results(['results/model1.json', 'results/model2.json'])

# Efficiency comparison
viz.plot_efficiency_comparison(save_path='plots/efficiency.png')

# Quality comparison
viz.plot_performance_comparison(save_path='plots/quality.png')

# Radar chart
viz.plot_radar_chart(
    metrics=['latency_ms_per_token', 'throughput_tokens_per_sec', 'average_score'],
    save_path='plots/radar.png'
)
```

### Export Results

```python
from analysis.exporter import ResultsExporter

exporter = ResultsExporter()
exporter.load_results(['results/model1.json', 'results/model2.json'])

# Export to multiple formats
exporter.export_all_formats(output_dir='exports', base_name='comparison')

# Or export specific format
exporter.export_to_csv('results.csv')
exporter.export_to_markdown('results.md')
exporter.export_to_latex('results.tex')
```

## Supported Models

### HuggingFace Models

```python
model = create_model_interface('huggingface')
model.load('meta-llama/Llama-2-7b-chat-hf')
```

### GPTQ (4-bit)

```python
model = create_model_interface('gptq')
model.load('TheBloke/Llama-2-7B-Chat-GPTQ')
```

### AWQ (4-bit)

```python
model = create_model_interface('awq')
model.load('TheBloke/Llama-2-7B-Chat-AWQ')
```

### HQQ (2/3/4/8-bit)

```python
model = create_model_interface('hqq')
model.load('meta-llama/Llama-2-7b-chat-hf', nbits=4, group_size=64)
```

## Metrics

### Efficiency Metrics

- **Latency**: ms/token (mean, std, min, max)
- **Throughput**: tokens/second
- **TTFT**: Time to first token
- **Memory**: Peak memory usage, model size, KV cache
- **FLOPs**: Computational operations per token
- **MFU**: Model FLOPs utilization
- **Energy**: Energy consumption per token

### Quality Metrics

- **Perplexity**: On WikiText or custom dataset
- **Task Accuracy**: Via lm-eval-harness
  - Commonsense reasoning (HellaSwag, PIQA, etc.)
  - World knowledge (TriviaQA, NQ)
  - Math (GSM8K, MATH)
  - Code (HumanEval, MBPP)
  - Aggregate benchmarks (MMLU, BBH)

## Analysis Tools

### Comparison

- Pairwise comparison with metrics
- Statistical significance testing
- Best/worst model identification
- Leaderboard creation

### Visualization

- Efficiency comparison plots
- Quality comparison plots
- Radar charts
- Custom metric plots

### Reporting

- HTML reports with plots
- Executive summaries
- Detailed metrics tables

### Export

- CSV for spreadsheets
- Markdown for documentation
- LaTeX for papers
- HTML for sharing

## CLI Commands

```bash
# Run benchmarks
python main.py --efficiency --quality

# Compare results
python -m analysis.comparator results/model1.json results/model2.json

# Generate visualizations
python -m analysis.visualizer results/*.json --dashboard

# Create report
python -m analysis.reporter results/*.json --output report.html

# Export results
python -m analysis.exporter results/*.json --output results.csv --format csv
```

## License

MIT License

## Citation

If you use this evaluation suite, please cite:

```bibtex
@software{llm_quant_eval,
  title={LLM Quantization Evaluation Suite},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-quant-eval}
}
```
