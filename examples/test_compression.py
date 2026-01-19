#!/usr/bin/env python3
"""
Test compression effectiveness for relevance storage.

Compares file sizes with different pickle protocols and compression settings.
"""

import torch
import tempfile
from pathlib import Path
import os


def create_mock_relevance_dict(vocab_size=151936, seq_len=32):
    """
    Create a mock relevance dictionary similar to real DLBacktrace output.
    """
    return {
        'embeddings': torch.randn(1, seq_len, 3584, dtype=torch.float32),
        'layer_0': torch.randn(1, seq_len, 3584, dtype=torch.float32),
        'layer_1': torch.randn(1, seq_len, 3584, dtype=torch.float32),
        'output_logits': torch.randn(1, seq_len, vocab_size, dtype=torch.float32),
        'attention_weights': torch.randn(1, 8, seq_len, seq_len, dtype=torch.float32),
    }


def test_compression_methods():
    """
    Test different compression configurations.
    """
    print("=" * 70)
    print("RELEVANCE STORAGE COMPRESSION TEST")
    print("=" * 70)
    
    # Create mock data
    print("\n1. Creating mock relevance data...")
    rel_dict = create_mock_relevance_dict()
    
    # Calculate uncompressed size in memory
    total_elements = sum(t.numel() for t in rel_dict.values())
    uncompressed_memory_mb = total_elements * 4 / (1024**2)  # float32 = 4 bytes
    print(f"   Total elements: {total_elements:,}")
    print(f"   Uncompressed memory (float32): {uncompressed_memory_mb:.2f} MB")
    
    # Test configurations
    configs = [
        {
            'name': 'Default (protocol=2, float32)',
            'dtype': torch.float32,
            'pickle_protocol': 2,
            'use_new_zip': False,
        },
        {
            'name': 'Protocol 2 + float16',
            'dtype': torch.float16,
            'pickle_protocol': 2,
            'use_new_zip': False,
        },
        {
            'name': 'Protocol 4 + float32 (NEW)',
            'dtype': torch.float32,
            'pickle_protocol': 4,
            'use_new_zip': True,
        },
        {
            'name': 'Protocol 4 + float16 (RECOMMENDED)',
            'dtype': torch.float16,
            'pickle_protocol': 4,
            'use_new_zip': True,
        },
        {
            'name': 'Protocol 5 + float16 (Best, Python 3.8+)',
            'dtype': torch.float16,
            'pickle_protocol': 5,
            'use_new_zip': True,
        },
    ]
    
    print("\n2. Testing compression configurations...\n")
    
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for config in configs:
            # Convert dtype
            converted_dict = {
                k: v.to(dtype=config['dtype']) 
                for k, v in rel_dict.items()
            }
            
            # Save with specific settings
            filepath = tmpdir / f"test_{config['pickle_protocol']}.pt"
            
            try:
                torch.save(
                    converted_dict,
                    filepath,
                    pickle_protocol=config['pickle_protocol'],
                    _use_new_zipfile_serialization=config['use_new_zip']
                )
                
                file_size_mb = os.path.getsize(filepath) / (1024**2)
                compression_ratio = uncompressed_memory_mb / file_size_mb
                
                results.append({
                    'name': config['name'],
                    'size_mb': file_size_mb,
                    'ratio': compression_ratio,
                })
                
                print(f"✓ {config['name']}")
                print(f"  File size: {file_size_mb:.2f} MB")
                print(f"  Compression ratio: {compression_ratio:.2f}x")
                print(f"  Savings vs uncompressed: {((1 - file_size_mb/uncompressed_memory_mb) * 100):.1f}%")
                print()
                
            except Exception as e:
                print(f"✗ {config['name']}: {e}\n")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline = results[0]['size_mb']
    recommended = results[3]['size_mb']
    
    print(f"\nBaseline (protocol 2, float32): {baseline:.2f} MB")
    print(f"Recommended (protocol 4, float16): {recommended:.2f} MB")
    print(f"**Improvement: {(baseline / recommended):.2f}x smaller** ({((1 - recommended/baseline) * 100):.1f}% reduction)")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION FOR NOTEBOOK")
    print("=" * 70)
    print("""
CACHE_CONFIG = {
    "policy": "disk",
    "dtype": "float16",
    "move_to_cpu": True,
    "dir": OUTPUT_DIR / "relevance_cache",
    
    # NEW: Enable compression (default: True)
    "use_compression": True,        # Use pickle protocol 4+
    "pickle_protocol": 4,            # Or 5 for Python 3.8+
}
""")


if __name__ == "__main__":
    test_compression_methods()
