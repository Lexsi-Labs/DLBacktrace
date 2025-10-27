"""
Quick Start Guide for DL-Backtrace Pipeline

This script demonstrates the simplest way to use the pipeline.
"""

from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig


def main():
    print("🚀 DL-Backtrace Pipeline - Quick Start")
    print("=" * 60)
    
    # Step 1: Create configuration
    # model_name is MANDATORY
    config = PipelineConfig(
        model_name="bert-sst2",  # REQUIRED - from ModelRegistry
        device="cpu",            # Use CPU for this example
        verbose=True             # Show detailed progress
    )
    
    # Step 2: Create pipeline instance
    pipeline = DLBacktracePipeline(config)
    
    # Step 3: Define your input text
    input_text = [
        "This movie is fantastic!",
        "I really didn't enjoy the meal."
    ]
    
    # Step 4: Run the pipeline
    results = pipeline.run(input_text)
    
    # Step 5: Explore results
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)
    print(f"Model: {results['model_name']}")
    print(f"Input sentences: {len(results['inputs'])}")
    print(f"Total time: {results['timing']['total_time']}s")
    print(f"  - Prediction: {results['timing']['predict_time']}s")
    print(f"  - Evaluation: {results['timing']['eval_time']}s")
    
    # Step 6: View relevance information
    pipeline.print_relevance_summary()
    
    # Results are automatically saved to 'backtrace_results/' directory
    print("\n✅ Pipeline completed successfully!")
    
    return results


if __name__ == "__main__":
    results = main()

