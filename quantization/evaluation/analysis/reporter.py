"""Report generator for evaluation results."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive HTML reports from evaluation results."""
    
    def __init__(self):
        self.results = []
        self.result_names = []
    
    def load_result(self, filepath: str, name: Optional[str] = None):
        """Load a single result file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {filepath}")
        
        with open(path, 'r') as f:
            result = json.load(f)
        
        result_name = name or path.stem
        self.results.append(result)
        self.result_names.append(result_name)
        
        logger.info(f"Loaded result: {result_name}")
    
    def load_results(self, filepaths: List[str], names: Optional[List[str]] = None):
        """Load multiple result files."""
        if names and len(names) != len(filepaths):
            raise ValueError("Length of names must match filepaths")
        
        for i, filepath in enumerate(filepaths):
            name = names[i] if names else None
            self.load_result(filepath, name)
    
    def generate_report(
        self,
        output_file: str,
        title: str = "Evaluation Report",
        include_plots: bool = True,
        plot_dir: Optional[str] = None
    ):
        """Generate comprehensive HTML report."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        if include_plots:
            if plot_dir is None:
                plot_dir = str(output_path.parent / 'plots')
            
            plot_paths = self._generate_plots(plot_dir)
        
        html = self._generate_html(title, plot_paths)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Report generated: {output_path}")
    
    def _generate_plots(self, plot_dir: str) -> Dict[str, str]:
        """Generate all visualization plots."""
        try:
            from analysis.visualizer import ResultsVisualizer
        except ImportError:
            logger.warning("Visualizer not available, skipping plots")
            return {}
        
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        viz = ResultsVisualizer()
        for result, name in zip(self.results, self.result_names):
            viz.results.append(result)
            viz.result_names.append(name)
        
        plot_paths = {}
        
        try:
            efficiency_path = str(Path(plot_dir) / 'efficiency.png')
            viz.plot_efficiency_comparison(save_path=efficiency_path)
            plot_paths['efficiency'] = efficiency_path
        except Exception as e:
            logger.warning(f"Failed to generate efficiency plot: {e}")
        
        try:
            quality_path = str(Path(plot_dir) / 'quality.png')
            viz.plot_quality_comparison(save_path=quality_path)
            plot_paths['quality'] = quality_path
        except Exception as e:
            logger.warning(f"Failed to generate quality plot: {e}")
        
        try:
            radar_path = str(Path(plot_dir) / 'radar.png')
            viz.plot_radar_chart(
                metrics=['latency_ms_per_token', 'throughput_tokens_per_sec', 'average_score'],
                save_path=radar_path
            )
            plot_paths['radar'] = radar_path
        except Exception as e:
            logger.warning(f"Failed to generate radar plot: {e}")
        
        return plot_paths
    
    def _generate_html(self, title: str, plot_paths: Dict[str, str]) -> str:
        """Generate HTML content."""
        html_parts = []
        
        html_parts.append(self._html_header(title))
        
        html_parts.append(f'''
        <div class="header">
            <h1>{self._escape_html(title)}</h1>
            <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p class="subtitle">Comparing {len(self.results)} model(s)</p>
        </div>
        ''')
        
        html_parts.append(self._generate_executive_summary())
        
        if plot_paths:
            html_parts.append(self._generate_plots_section(plot_paths))
        
        html_parts.append(self._generate_detailed_results())
        
        if len(self.results) > 1:
            html_parts.append(self._generate_comparison_section())
        
        html_parts.append(self._html_footer())
        
        return '\n'.join(html_parts)
    
    def _html_header(self, title: str) -> str:
        """Generate HTML header with CSS."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._escape_html(title)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #4CAF50;
        }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }}
        h2 {{ color: #34495e; margin: 30px 0 15px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }}
        h3 {{ color: #5a6c7d; margin: 20px 0 10px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        .subtitle {{ color: #95a5a6; margin-top: 10px; }}
        
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 6px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        .metric-value {{
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: bold;
        }}
        .metric-unit {{
            color: #95a5a6;
            font-size: 0.7em;
            margin-left: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        tr:hover {{ background-color: #f8f9fa; }}
        tr:last-child td {{ border-bottom: none; }}
        
        .best {{ background-color: #d4edda !important; font-weight: bold; }}
        .worst {{ background-color: #f8d7da !important; }}
        
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .comparison-row {{
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .improvement {{ color: #28a745; }}
        .regression {{ color: #dc3545; }}
    </style>
</head>
<body>
<div class="container">
'''
    
    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return '''
</div>
</body>
</html>
'''
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        html = ['<div class="section">', '<h2>Executive Summary</h2>']
        
        html.append('<div class="metric-grid">')
        
        for name, result in zip(self.result_names, self.results):
            html.append(f'<div><h3>{self._escape_html(name)}</h3>')
            
            metrics = [
                ('Latency', 'latency_ms_per_token', 'ms/tok'),
                ('Throughput', 'throughput_tokens_per_sec', 'tok/s'),
                ('Perplexity', 'perplexity', ''),
                ('Avg Score', 'average_score', '%')
            ]
            
            for label, key, unit in metrics:
                value = self._find_metric_value(result, key)
                if value is not None:
                    if key == 'average_score':
                        value *= 100
                    html.append(f'''
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">
                            {value:.2f}<span class="metric-unit">{unit}</span>
                        </div>
                    </div>
                    ''')
            
            html.append('</div>')
        
        html.append('</div>')
        html.append('</div>')
        
        return '\n'.join(html)
    
    def _generate_plots_section(self, plot_paths: Dict[str, str]) -> str:
        """Generate plots section."""
        html = ['<div class="section">', '<h2>Visualizations</h2>']
        
        plot_titles = {
            'efficiency': 'Efficiency Comparison',
            'quality': 'Quality Comparison',
            'radar': 'Overall Performance Radar'
        }
        
        for plot_type, plot_path in plot_paths.items():
            title = plot_titles.get(plot_type, plot_type.title())
            rel_path = Path(plot_path).name
            html.append(f'''
            <div class="plot-container">
                <h3>{title}</h3>
                <img src="plots/{rel_path}" alt="{title}">
            </div>
            ''')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _generate_detailed_results(self) -> str:
        """Generate detailed results tables."""
        html = ['<div class="section">', '<h2>Detailed Results</h2>']
        
        sections = [
            ('Efficiency', ['latency_ms_per_token', 'throughput_tokens_per_sec',
                          'peak_memory_mb', 'energy_per_token_mj']),
            ('Quality', ['perplexity', 'average_score']),
        ]
        
        for section_name, metrics in sections:
            html.append(f'<h3>{section_name}</h3>')
            html.append(self._create_metrics_table(metrics))
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _create_metrics_table(self, metrics: List[str]) -> str:
        """Create HTML table for metrics."""
        html = ['<table>', '<thead><tr><th>Metric</th>']
        
        for name in self.result_names:
            html.append(f'<th>{self._escape_html(name)}</th>')
        
        html.append('</tr></thead><tbody>')
        
        for metric in metrics:
            values = []
            for result in self.results:
                value = self._find_metric_value(result, metric)
                values.append(value)
            
            if all(v is None for v in values):
                continue
            
            html.append(f'<tr><td><strong>{metric}</strong></td>')
            
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                best_val = max(numeric_values)
                worst_val = min(numeric_values)
            else:
                best_val = worst_val = None
            
            for value in values:
                cell_class = ''
                if value == best_val and best_val != worst_val:
                    cell_class = ' class="best"'
                elif value == worst_val and best_val != worst_val:
                    cell_class = ' class="worst"'
                
                if value is None:
                    formatted = 'N/A'
                elif isinstance(value, float):
                    formatted = f'{value:.4f}'
                else:
                    formatted = str(value)
                
                html.append(f'<td{cell_class}>{formatted}</td>')
            
            html.append('</tr>')
        
        html.append('</tbody></table>')
        return '\n'.join(html)
    
    def _generate_comparison_section(self) -> str:
        """Generate pairwise comparison section."""
        html = ['<div class="section">', '<h2>Model Comparisons</h2>']
        
        for i in range(len(self.results) - 1):
            html.append(f'<h3>{self.result_names[i+1]} vs {self.result_names[i]}</h3>')
            html.append(self._create_comparison_content(i, i+1))
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _create_comparison_content(self, baseline_idx: int, comp_idx: int) -> str:
        """Create comparison content."""
        html = []
        
        metrics = ['latency_ms_per_token', 'throughput_tokens_per_sec', 'average_score']
        
        for metric in metrics:
            baseline_val = self._find_metric_value(self.results[baseline_idx], metric)
            comp_val = self._find_metric_value(self.results[comp_idx], metric)
            
            if baseline_val is None or comp_val is None:
                continue
            
            diff = comp_val - baseline_val
            pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            is_improvement = diff > 0 if 'score' in metric or 'throughput' in metric else diff < 0
            
            symbol = '↑' if diff > 0 else '↓'
            css_class = 'improvement' if is_improvement else 'regression'
            
            html.append(f'''
            <div class="comparison-row">
                <span><strong>{metric}</strong></span>
                <span class="{css_class}">
                    {symbol} {abs(pct_change):.2f}%
                    ({baseline_val:.4f} -> {comp_val:.4f})
                </span>
            </div>
            ''')
        
        return '\n'.join(html)
    
    def _find_metric_value(self, result: Dict, metric: str) -> Any:
        """Find a metric value in nested dictionary."""
        if metric in result:
            return result[metric]
        
        for key, value in result.items():
            if isinstance(value, dict) and metric in value:
                return value[metric]
        
        return None
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (str(text)
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


def main():
    """CLI interface for reporter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('files', nargs='+', help='JSON result files')
    parser.add_argument('--names', nargs='+', help='Custom names for results')
    parser.add_argument('--output', required=True, help='Output HTML file')
    parser.add_argument('--title', default='Evaluation Report', help='Report title')
    parser.add_argument('--no-plots', action='store_true', help='Disable plots')
    parser.add_argument('--plot-dir', help='Directory for plot images')
    
    args = parser.parse_args()
    
    reporter = ReportGenerator()
    reporter.load_results(args.files, args.names)
    reporter.generate_report(
        output_file=args.output,
        title=args.title,
        include_plots=not args.no_plots,
        plot_dir=args.plot_dir
    )
    
    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()