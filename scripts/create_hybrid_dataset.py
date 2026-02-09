#!/usr/bin/env python
"""
Hybrid Dataset Creator
Combines curated (70%) and InstructLab synthetic (30%) data with schema validation
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

# JSON Schema for validation
DATASET_SCHEMA = {
    "required_fields": ["instruction", "output"],
    "optional_fields": ["metadata"],
    "instruction_min_length": 10,
    "output_min_length": 20,
    "instruction_max_length": 1000,
    "output_max_length": 5000
}

def validate_schema(sample: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate sample against JSON schema
    Returns (is_valid, error_message)
    """
    # Check required fields
    for field in DATASET_SCHEMA["required_fields"]:
        if field not in sample:
            return False, f"Missing required field: {field}"
    
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    
    # Validate lengths
    if len(instruction) < DATASET_SCHEMA["instruction_min_length"]:
        return False, f"Instruction too short: {len(instruction)} < {DATASET_SCHEMA['instruction_min_length']}"
    
    if len(output) < DATASET_SCHEMA["output_min_length"]:
        return False, f"Output too short: {len(output)} < {DATASET_SCHEMA['output_min_length']}"
    
    if len(instruction) > DATASET_SCHEMA["instruction_max_length"]:
        return False, f"Instruction too long: {len(instruction)} > {DATASET_SCHEMA['instruction_max_length']}"
    
    if len(output) > DATASET_SCHEMA["output_max_length"]:
        return False, f"Output too long: {len(output)} > {DATASET_SCHEMA['output_max_length']}"
    
    # Validate types
    if not isinstance(instruction, str) or not isinstance(output, str):
        return False, "instruction and output must be strings"
    
    return True, None

def load_curated_data(filepath: str = "data/curated_seed_data.jsonl") -> List[Dict]:
    """Load and validate curated seed data"""
    samples = []
    invalid_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                is_valid, error = validate_schema(sample)
                if is_valid:
                    if 'metadata' not in sample:
                        sample['metadata'] = {}
                    sample['metadata']['source'] = 'curated'
                    samples.append(sample)
                else:
                    invalid_count += 1
                    print(f"  ⚠️  Line {line_num}: {error}")
            except json.JSONDecodeError:
                invalid_count += 1
                print(f"  ❌ Line {line_num}: Invalid JSON")
    
    print(f"  Loaded {len(samples)} curated samples ({invalid_count} invalid)")
    return samples

def load_instructlab_data(directory: str = "data/instructlab_real") -> List[Dict]:
    """Load and convert InstructLab synthetic data (Checks ONLY the latest file to prevent duplication)"""
    samples = []
    invalid_count = 0
    
    data_path = Path(directory)
    if not data_path.exists():
        print(f"Warning: {directory} does not exist.")
        return []
    
    # Find all potential files with their sortable keys
    candidate_files = []
    
    # 1. Look for preprocessed files (highest priority) -> Sort by folder name timestamp
    preproc_files = list(data_path.glob("preprocessed_*/compositional_skills_qlorax.jsonl"))
    for f in preproc_files:
        candidate_files.append((f, f.parent.name)) 
        
    # 2. Look for test files if no preprocessed -> Sort by filename timestamp
    if not candidate_files:
        test_files = list(data_path.glob("test_*.jsonl"))
        for f in test_files:
            candidate_files.append((f, f.name))
            
    # 3. Fallback to any .jsonl -> Sort by file mtime
    if not candidate_files:
        other_files = [f for f in data_path.glob("*.jsonl") if not f.name.startswith("test_")]
        for f in other_files:
            candidate_files.append((f, str(f.stat().st_mtime)))
            
    if not candidate_files:
        print("  No InstructLab data files found.")
        return []
        
    # Sort descending (latest first) and pick top 1
    candidate_files.sort(key=lambda x: x[1], reverse=True)
    selected_file = candidate_files[0][0]
    print(f"  Using latest dataset file: {selected_file}")
    
    # Process ONLY the selected file
    with open(selected_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # Convert InstructLab format to standard format
                if 'user' in data and 'assistant' in data:
                    sample = {
                        'instruction': data['user'],
                        'output': data['assistant'],
                        'metadata': {'source': 'instructlab_synthetic', 'model': 'llama-3-1-70b'}
                    }
                elif 'seed_question' in data and 'seed_response' in data:
                    sample = {
                        'instruction': data['seed_question'],
                        'output': data['seed_response'],
                        'metadata': {'source': 'instructlab_synthetic', 'task': data.get('task_description', '')}
                    }
                else:
                    # Already in standard format
                    sample = data
                    if 'metadata' not in sample:
                        sample['metadata'] = {}
                    sample['metadata']['source'] = 'instructlab_synthetic'
                
                # Validate
                is_valid, error = validate_schema(sample)
                if is_valid:
                    samples.append(sample)
                else:
                    invalid_count += 1
            
            except (json.JSONDecodeError, KeyError):
                invalid_count += 1
    
    print(f"  Loaded {len(samples)} InstructLab samples ({invalid_count} invalid)")
    return samples

def create_hybrid_dataset(
    curated_ratio: float = 0.7,
    synthetic_ratio: float = 0.3,
    output_dir: str = "data/variants",
    random_seed: int = 42
):
    """
    Create hybrid dataset with 70% curated + 30% synthetic
    Also creates curated-only and synthetic-only variants
    """
    random.seed(random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("HYBRID DATASET CREATION")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading curated data...")
    curated_samples = load_curated_data()
    
    print("\n[2/4] Loading InstructLab synthetic data...")
    synthetic_samples = load_instructlab_data()
    
    if not curated_samples and not synthetic_samples:
        print("\n❌ ERROR: No data found!")
        return
    
    # Create variants
    print(f"\n[3/4] Creating dataset variants...")
    
    variants = {}
    
    # Variant 1: Curated-only
    if curated_samples:
        variants['curated_only'] = curated_samples.copy()
        random.shuffle(variants['curated_only'])
    
    # Variant 2: Synthetic-only
    if synthetic_samples:
        variants['synthetic_only'] = synthetic_samples.copy()
        random.shuffle(variants['synthetic_only'])
    
    # Variant 3: Hybrid (70/30)
    if curated_samples and synthetic_samples:
        # Calculate target counts
        total_target = max(len(curated_samples), len(synthetic_samples)) * 2
        curated_target = int(total_target * curated_ratio)
        synthetic_target = int(total_target * synthetic_ratio)
        
        # Sample with replacement if needed
        curated_selected = random.choices(curated_samples, k=curated_target)
        synthetic_selected = random.choices(synthetic_samples, k=synthetic_target)
        
        variants['hybrid_70_30'] = curated_selected + synthetic_selected
        random.shuffle(variants['hybrid_70_30'])
    
    # Save variants
    print(f"\n[4/4] Saving dataset variants...")
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'curated_ratio': curated_ratio,
        'synthetic_ratio': synthetic_ratio,
        'random_seed': random_seed,
        'schema': DATASET_SCHEMA,
        'variants': {}
    }
    
    for variant_name, variant_data in variants.items():
        variant_file = output_path / f"{variant_name}.jsonl"
        
        with open(variant_file, 'w', encoding='utf-8') as f:
            for sample in variant_data:
                f.write(json.dumps(sample) + '\n')
        
        # Calculate stats
        curated_count = sum(1 for s in variant_data if s.get('metadata', {}).get('source') == 'curated')
        synthetic_count = sum(1 for s in variant_data if s.get('metadata', {}).get('source') == 'instructlab_synthetic')
        
        metadata['variants'][variant_name] = {
            'total_samples': len(variant_data),
            'curated_samples': curated_count,
            'synthetic_samples': synthetic_count,
            'curated_percentage': round(curated_count / len(variant_data) * 100, 1) if variant_data else 0,
            'file': str(variant_file)
        }
        
        print(f"  ✅ {variant_name}: {len(variant_data)} samples ({curated_count} curated, {synthetic_count} synthetic)")
    
    # Save metadata
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DATASET VARIANTS CREATED")
    print(f"{'='*70}")
    print(f"Location: {output_path}")
    print(f"\nVariants:")
    for name, info in metadata['variants'].items():
        print(f"  • {name}: {info['total_samples']} samples ({info['curated_percentage']}% curated)")
    
    return metadata

if __name__ == "__main__":
    import sys
    
    # Parse args
    curated_ratio = 0.7
    synthetic_ratio = 0.3
    
    if len(sys.argv) > 1:
        try:
            curated_ratio = float(sys.argv[1])
            synthetic_ratio = 1.0 - curated_ratio
        except ValueError:
            print("Usage: python create_hybrid_dataset.py [curated_ratio]")
            print("Example: python create_hybrid_dataset.py 0.7")
            sys.exit(1)
    
    create_hybrid_dataset(curated_ratio, synthetic_ratio)
