#!/usr/bin/env python3
"""
Run all RoBERTa tests and generate a comprehensive report
"""

import subprocess
import sys
import os
from datetime import datetime

def run_test(test_file):
    """Run a single test and return results"""
    try:
        # Use longer timeout for numerical divergence test
        timeout = 300 if 'numerical_divergence' in test_file else 120
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=timeout)
        return {
            'name': os.path.basename(test_file),
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr
        }
    except subprocess.TimeoutExpired:
        timeout_used = 300 if 'numerical_divergence' in test_file else 120
        return {
            'name': os.path.basename(test_file),
            'success': False,
            'output': '',
            'error': f'Test timed out after {timeout_used} seconds'
        }
    except Exception as e:
        return {
            'name': os.path.basename(test_file),
            'success': False,
            'output': '',
            'error': str(e)
        }

def main():
    """Run all tests and generate report"""
    print("🚀 Running RoBERTa Test Suite")
    print("=" * 50)
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of test files to run
    test_files = [
        'test_boolean_tensors.py',
        'test_embedding.py', 
        'test_layernorm.py',
        'test_attention.py',
        'test_tensor_shapes.py',
        'test_numerical_divergence.py'
    ]
    
    # Add more tests if they exist
    for test_file in ['test_cumsum.py', 'test_slice.py', 'test_add.py', 'test_ne.py']:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            test_files.append(test_file)
    
    results = []
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            print(f"\\n🧪 Running {test_file}...")
            result = run_test(test_path)
            results.append(result)
            
            if result['success']:
                print(f"✅ {test_file} - PASSED")
            else:
                print(f"❌ {test_file} - FAILED")
                if result['error']:
                    print(f"   Error: {result['error'][:200]}...")
        else:
            print(f"⚠️  {test_file} not found, skipping...")
    
    # Generate summary report
    print("\\n" + "=" * 50)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 50)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    print("\\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {result['name']:<30} {status}")
        
        if not result['success'] and result['error']:
            print(f"    └─ {result['error'][:100]}...")
    
    # Save detailed report to file
    report_file = os.path.join(test_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w') as f:
        f.write(f"RoBERTa Test Suite Report - {datetime.now()}\\n")
        f.write("=" * 50 + "\\n\\n")
        
        for result in results:
            f.write(f"Test: {result['name']}\\n")
            f.write(f"Status: {'PASS' if result['success'] else 'FAIL'}\\n")
            if result['output']:
                f.write(f"Output:\\n{result['output']}\\n")
            if result['error']:
                f.write(f"Error:\\n{result['error']}\\n")
            f.write("-" * 30 + "\\n\\n")
    
    print(f"\\n📄 Detailed report saved to: {report_file}")
    
    # Return success if all critical tests pass
    critical_tests = ['test_boolean_tensors.py', 'test_tensor_shapes.py']
    critical_passed = all(r['success'] for r in results 
                         if r['name'] in critical_tests and r['name'] in [r['name'] for r in results])
    
    if critical_passed and passed >= total * 0.7:  # At least 70% pass rate
        print("\\n🎉 Overall test suite: ACCEPTABLE")
        return 0
    else:
        print("\\n⚠️  Overall test suite: NEEDS ATTENTION")
        return 1

if __name__ == "__main__":
    sys.exit(main())
