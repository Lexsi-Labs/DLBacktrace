"""
Example usage of DL-Backtrace Pipeline

This example demonstrates how to use the DL-Backtrace pipeline
to run explainability analysis on PyTorch models.
"""

from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig


def basic_example():
    """Basic pipeline usage example"""
    print("="*60)
    print("DL-Backtrace Pipeline - Basic Example")
    print("="*60)
    
    # Create configuration
    config = PipelineConfig(
        model_name="bert-sst2",  # Mandatory field
        device="cpu",
        batch_size=2,
        max_length=128,
        verbose=True,
        save_results=True
    )
    
    # Create pipeline
    pipeline = DLBacktracePipeline(config)
    
    # Define input text
    input_text = [
        "This movie is fantastic!",
        "I really didn't enjoy the meal."
    ]
    
    # Run the pipeline
    results = pipeline.run(input_text)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {results['model_name']}")
    print(f"Total time: {results['timing']['total_time']}s")
    print(f"Predict time: {results['timing']['predict_time']}s")
    print(f"Eval time: {results['timing']['eval_time']}s")
    print("="*60)
    
    # Print relevance summary
    pipeline.print_relevance_summary()
    
    return results


def advanced_example():
    """Advanced pipeline usage with custom configuration"""
    print("\n" + "="*60)
    print("DL-Backtrace Pipeline - Advanced Example")
    print("="*60)
    
    # Create configuration with custom parameters
    config = PipelineConfig(
        model_name="bert-sst2",
        device="cpu",
        batch_size=1,
        max_length=256,
        verbose=True,
        strict_cpu=True,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        task="binary-classification",
        temperature=1.0,
        save_results=True,
        save_visualization=True,
        output_dir="custom_results"
    )
    
    # Create pipeline
    pipeline = DLBacktracePipeline(config)
    
    # Run with custom input
    input_text = [
        "The service was excellent and the food was delicious."
    ]
    
    results = pipeline.run(input_text)
    
    return results


def multi_model_example():
    """Example showing how to run multiple models"""
    print("\n" + "="*60)
    print("DL-Backtrace Pipeline - Multi-Model Example")
    print("="*60)
    
    models = ["bert-sst2", "roberta-base", "distilbert"]
    
    results = {}
    
    for model_name in models:
        print(f"\n🔄 Processing model: {model_name}")
        
        try:
            config = PipelineConfig(
                model_name=model_name,
                device="cpu",
                verbose=False,  # Less verbose for batch processing
                save_results=True
            )
            
            pipeline = DLBacktracePipeline(config)
            
            input_text = ["This is a test sentence."]
            model_results = pipeline.run(input_text)
            
            results[model_name] = model_results
            
            print(f"✅ Completed: {model_name}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    for model_name, result in results.items():
        if "error" in result:
            print(f"{model_name}: ❌ {result['error']}")
        else:
            print(f"{model_name}: ✅ {result['timing']['total_time']}s")
    
    return results


if __name__ == "__main__":
    # Run basic example
    results = basic_example()
    
    # Uncomment to run advanced example
    # advanced_results = advanced_example()
    
    # Uncomment to run multi-model example
    # multi_results = multi_model_example()

