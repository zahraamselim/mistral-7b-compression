{
  "model": {
    "model_path": "meta-llama/Llama-2-7b-chat-hf",
    "interface_type": "huggingface",
    "torch_dtype": "float16",
    "device_map": "auto",
    "trust_remote_code": false,
    "quantization": null
  },
  
  "benchmarks": {
    "efficiency": {
      "num_warmup": 3,
      "num_runs": 10,
      "max_new_tokens": 128,
      "prompts": [
        "The capital of France is",
        "Artificial intelligence is defined as",
        "The main benefit of renewable energy sources is",
        "In machine learning, the term 'overfitting' refers to",
        "Quantum computing differs from classical computing because",
        "The theory of relativity states that",
        "Neural networks are computational models inspired by",
        "Climate change is primarily caused by"
      ],
      "measure_prefill_decode": true,
      "measure_batch_throughput": false,
      "batch_sizes": [1, 2, 4, 8],
      "estimate_kv_cache": true
    },
    
    "quality": {
      "perplexity": {
        "enabled": true,
        "dataset": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "num_samples": 100,
        "max_length": 512,
        "stride": null,
        "batch_size": 1
      },
      
      "tasks": {
        "enabled": false,
        "batch_size": 1,
        "task_list": {
          "hellaswag": {
            "enabled": false,
            "num_fewshot": 0,
            "limit": null
          },
          "winogrande": {
            "enabled": false,
            "num_fewshot": 0,
            "limit": null
          },
          "piqa": {
            "enabled": false,
            "num_fewshot": 0,
            "limit": null
          },
          "arc_easy": {
            "enabled": false,
            "num_fewshot": 0,
            "limit": null
          },
          "arc_challenge": {
            "enabled": false,
            "num_fewshot": 0,
            "limit": null
          },
          "mmlu": {
            "enabled": false,
            "num_fewshot": 5,
            "limit": null
          }
        }
      }
    }
  },
  
  "output_dir": "./results"
}