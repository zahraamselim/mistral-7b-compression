"""
Research-Grade RAG Calibration Dataset Generator
Creates structured calibration data for quantization with RAG-specific patterns
"""

import json
import random
from datasets import load_dataset
from typing import List, Dict
import numpy as np

class RAGCalibrationDataset:
    def __init__(self, n_samples=128, seed=42):
        """
        Initialize RAG calibration dataset generator
        
        Args:
            n_samples: Total calibration samples (default 128 per AWQ/GPTQ standards)
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Distribution per thesis specification
        self.distribution = {
            'short': 32,      # 256-512 tokens: single doc retrieval
            'medium': 40,     # 1024-2048 tokens: 2-3 docs
            'long': 32,       # 3072-4096 tokens: 5-7 docs
            'multi_hop': 24   # Complex reasoning: 8-10 docs
        }
        
    def load_source_datasets(self):
        """Load real datasets as specified in thesis"""
        print("Loading source datasets...")
        
        # For question-answer pairs
        self.squad = load_dataset("squad_v2", split="train")
        
        # For multi-hop reasoning
        self.hotpot = load_dataset("hotpot_qa", "distractor", split="train")
        
        # For documents (using Wikitext - simpler and more reliable)
        self.wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
        self.wiki = self.wiki.filter(lambda x: len(x["text"]) > 200)
        
        print(f"Loaded: {len(self.squad)} SQuAD, {len(self.hotpot)} HotpotQA, {len(self.wiki)} Wikitext docs")
        
    def create_short_context_samples(self) -> List[str]:
        """
        Short context: 256-512 tokens
        Single document retrieval, direct QA
        """
        samples = []
        
        for i in range(self.distribution['short']):
            # Get question and context from SQuAD
            item = self.squad[random.randint(0, len(self.squad)-1)]
            
            query = item['question']
            context = item['context']
            answer = item['answers']['text'][0] if item['answers']['text'] else "Unanswerable"
            
            # Format as RAG prompt
            formatted = f"""[QUERY]: {query}

[RETRIEVED DOCUMENTS]:
Document 1: {context}

[EXPECTED ANSWER]: {answer}"""
            
            samples.append(formatted)
            
        return samples
    
    def create_medium_context_samples(self) -> List[str]:
        """
        Medium context: 1024-2048 tokens
        2-3 documents requiring comparison/synthesis
        """
        samples = []
        
        for i in range(self.distribution['medium']):
            # Get base question
            item = self.squad[random.randint(0, len(self.squad)-1)]
            query = item['question']
            doc1 = item['context']
            
            # Add 1-2 related documents from Wikitext
            n_extra_docs = random.randint(1, 2)
            extra_docs = []
            
            for j in range(n_extra_docs):
                wiki_item = self.wiki[random.randint(0, len(self.wiki)-1)]
                # Truncate to reasonable length
                wiki_text = wiki_item['text'].strip()
                if wiki_text:
                    extra_docs.append(wiki_text[:800])
            
            # Format
            doc_section = "\n\n".join([
                f"Document {idx+1}: {doc}" 
                for idx, doc in enumerate([doc1] + extra_docs)
            ])
            
            answer = item['answers']['text'][0] if item['answers']['text'] else "Information requires synthesis across documents"
            
            formatted = f"""[QUERY]: {query}

[RETRIEVED DOCUMENTS]:
{doc_section}

[EXPECTED ANSWER]: {answer}"""
            
            samples.append(formatted)
            
        return samples
    
    def create_long_context_samples(self) -> List[str]:
        """
        Long context: 3072-4096 tokens
        5-7 documents requiring complex synthesis
        """
        samples = []
        
        for i in range(self.distribution['long']):
            # Use HotpotQA for complex questions
            item = self.hotpot[random.randint(0, len(self.hotpot)-1)]
            query = item['question']
            
            # Get supporting documents
            supporting_docs = []
            for context in item['context']['sentences'][:7]:  # Up to 7 paragraphs
                supporting_docs.append(' '.join(context))
            
            # Add distractors from Wikitext
            n_distractors = max(0, 7 - len(supporting_docs))
            for j in range(n_distractors):
                wiki_item = self.wiki[random.randint(0, len(self.wiki)-1)]
                wiki_text = wiki_item['text'].strip()
                if wiki_text:
                    supporting_docs.append(wiki_text[:600])
            
            # Shuffle to make retrieval harder
            random.shuffle(supporting_docs)
            
            doc_section = "\n\n".join([
                f"Document {idx+1}: {doc}" 
                for idx, doc in enumerate(supporting_docs)
            ])
            
            answer = item['answer']
            
            formatted = f"""[QUERY]: {query}

[RETRIEVED DOCUMENTS]:
{doc_section}

[EXPECTED ANSWER]: {answer}"""
            
            samples.append(formatted)
            
        return samples
    
    def create_multi_hop_samples(self) -> List[str]:
        """
        Multi-hop: 8-10 documents
        Requires temporal ordering, comparison, aggregation
        """
        samples = []
        
        for i in range(self.distribution['multi_hop']):
            # Use HotpotQA comparison/bridge questions
            item = self.hotpot[random.randint(0, len(self.hotpot)-1)]
            
            # Ensure it's a comparison or bridge type
            while item['type'] not in ['comparison', 'bridge']:
                item = self.hotpot[random.randint(0, len(self.hotpot)-1)]
            
            query = item['question']
            
            # Get all available context
            all_contexts = []
            for context in item['context']['sentences']:
                all_contexts.append(' '.join(context))
            
            # Pad with Wikitext to reach 8-10 docs
            target_docs = random.randint(8, 10)
            while len(all_contexts) < target_docs:
                wiki_item = self.wiki[random.randint(0, len(self.wiki)-1)]
                wiki_text = wiki_item['text'].strip()
                if wiki_text:
                    all_contexts.append(wiki_text[:500])
            
            # Shuffle
            random.shuffle(all_contexts)
            all_contexts = all_contexts[:target_docs]
            
            doc_section = "\n\n".join([
                f"Document {idx+1}: {doc}" 
                for idx, doc in enumerate(all_contexts)
            ])
            
            answer = item['answer']
            
            formatted = f"""[QUERY]: {query}

[RETRIEVED DOCUMENTS]:
{doc_section}

[EXPECTED ANSWER]: {answer}"""
            
            samples.append(formatted)
            
        return samples
    
    def generate_dataset(self) -> List[str]:
        """Generate complete calibration dataset"""
        print(f"\nGenerating {self.n_samples} RAG calibration samples...")
        
        self.load_source_datasets()
        
        all_samples = []
        
        # Generate each category
        print(f"Creating {self.distribution['short']} short-context samples...")
        all_samples.extend(self.create_short_context_samples())
        
        print(f"Creating {self.distribution['medium']} medium-context samples...")
        all_samples.extend(self.create_medium_context_samples())
        
        print(f"Creating {self.distribution['long']} long-context samples...")
        all_samples.extend(self.create_long_context_samples())
        
        print(f"Creating {self.distribution['multi_hop']} multi-hop samples...")
        all_samples.extend(self.create_multi_hop_samples())
        
        # Shuffle final dataset
        random.shuffle(all_samples)
        
        # Validate length distribution
        self._print_statistics(all_samples)
        
        return all_samples
    
    def _print_statistics(self, samples: List[str]):
        """Print dataset statistics"""
        lengths = [len(s.split()) for s in samples]
        
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {len(samples)}")
        print(f"Token count (approx):")
        print(f"  Mean: {np.mean(lengths):.0f} tokens")
        print(f"  Median: {np.median(lengths):.0f} tokens")
        print(f"  Min: {np.min(lengths)} tokens")
        print(f"  Max: {np.max(lengths)} tokens")
        print(f"  Std: {np.std(lengths):.0f} tokens")
        
        # Length distribution
        bins = [0, 512, 1024, 2048, 4096, float('inf')]
        labels = ['<512', '512-1K', '1K-2K', '2K-4K', '>4K']
        
        print("\nLength distribution:")
        for i in range(len(bins)-1):
            count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
            pct = 100 * count / len(lengths)
            print(f"  {labels[i]}: {count} samples ({pct:.1f}%)")
    
    def save_dataset(self, samples: List[str], output_path: str = "rag_calibration.json"):
        """Save dataset to JSON"""
        with open(output_path, 'w') as f:
            json.dump({
                'samples': samples,
                'metadata': {
                    'n_samples': len(samples),
                    'seed': self.seed,
                    'distribution': self.distribution,
                    'format': 'RAG-formatted calibration data',
                    'sources': ['SQuAD v2', 'HotpotQA', 'Wikitext-103']
                }
            }, f, indent=2)
        
        print(f"\nSaved to {output_path}")


# Usage example
if __name__ == "__main__":
    generator = RAGCalibrationDataset(n_samples=128, seed=42)
    calibration_samples = generator.generate_dataset()
    generator.save_dataset(calibration_samples)
    
    print("\nRAG calibration dataset ready for quantization!")