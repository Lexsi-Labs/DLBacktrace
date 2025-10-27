"""
DL-Backtrace Pipeline - Simple Entry Point

A simple script to run the DL-Backtrace pipeline with minimal configuration.
Requires 'model_name' as a mandatory parameter.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig


def main():
    """Main entry point for DL-Backtrace Pipeline"""
    
    parser = argparse.ArgumentParser(
        description='DL-Backtrace Pipeline: Explainability analysis for PyTorch models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with mandatory model_name
  python run_dlbacktrace_pipeline.py --model-name bert-sst2 --input "This movie is great!"
  
  # With custom device and batch size
  python run_dlbacktrace_pipeline.py --model-name llama-3.2-1b --input "Hello world" --device cuda --batch-size 1
  
  # With verbose debugging
  python run_dlbacktrace_pipeline.py --model-name roberta-base --input "Test" --verbose --debug
  
  # Save visualizations
  python run_dlbacktrace_pipeline.py --model-name bert-sst2 --input "Test" --save-viz
        """
    )
    
    # Mandatory arguments
    parser.add_argument('--model-name', required=True, type=str,
                       help='Model name (MANDATORY). Must be from ModelRegistry: bert-sst2, roberta-base, distilbert, llama-3.2-1b, phi-4, gpt2')
    
    parser.add_argument('--input', required=True, type=str, nargs='+',
                       help='Input text to analyze. Can be one or more sentences.')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    
    # Input configuration
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    
    # DL-Backtrace configuration
    parser.add_argument('--mode', type=str, default='default',
                       help='Relevance propagation mode (default: default)')
    
    parser.add_argument('--multiplier', type=float, default=100.0,
                       help='Relevance multiplier (default: 100.0)')
    
    parser.add_argument('--scaler', type=float, default=1.0,
                       help='Relevance scaler (default: 1.0)')
    
    parser.add_argument('--thresholding', type=float, default=0.5,
                       help='Relevance thresholding (default: 0.5)')
    
    parser.add_argument('--task', type=str, default='binary-classification',
                       choices=['binary-classification', 'multi-class-classification', 'regression', 'generation'],
                       help='Task type (default: binary-classification)')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature scaling (default: 1.0)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='backtrace_results',
                       help='Output directory (default: backtrace_results)')
    
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save results to disk (default: True)')
    
    parser.add_argument('--save-viz', action='store_true', default=False,
                       help='Save visualizations (default: False)')
    
    # Debug configuration
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    parser.add_argument('--strict-cpu', action='store_true', default=True,
                       help='Use strict CPU determinism (default: True)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        model_name=args.model_name,  # Mandatory
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mode=args.mode,
        multiplier=args.multiplier,
        scaler=args.scaler,
        thresholding=args.thresholding,
        task=args.task,
        temperature=args.temperature,
        save_results=args.save_results,
        save_visualization=args.save_viz,
        output_dir=args.output_dir,
        verbose=args.verbose,
        debug=args.debug,
        strict_cpu=args.strict_cpu
    )
    
    # Display configuration
    print("="*70)
    print("DL-Backtrace Pipeline Configuration")
    print("="*70)
    print(f"Model Name:     {config.model_name}")
    print(f"Device:         {config.device}")
    print(f"Batch Size:     {config.batch_size}")
    print(f"Max Length:     {config.max_length}")
    print(f"Task:           {config.task}")
    print(f"Mode:           {config.mode}")
    print(f"Multiplier:     {config.multiplier}")
    print(f"Temperature:    {config.temperature}")
    print(f"Save Results:   {config.save_results}")
    print(f"Save Viz:       {config.save_visualization}")
    print(f"Output Dir:     {config.output_dir}")
    print(f"Verbose:        {config.verbose}")
    print(f"Debug:          {config.debug}")
    print("="*70)
    
    # Create pipeline
    try:
        pipeline = DLBacktracePipeline(config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nAvailable models:")
        from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import ModelRegistry
        models = ModelRegistry.list_models()
        for name, description in models.items():
            print(f"  - {name}: {description}")
        sys.exit(1)
    
    # Run pipeline
    print(f"\n🔄 Running pipeline with {len(args.input)} input(s)...")
    print("Input:", args.input)
    print()
    
    try:
        results = pipeline.run(args.input)
        
        # Display results summary
        print("\n" + "="*70)
        print("📊 RESULTS SUMMARY")
        print("="*70)
        print(f"Model:              {results['model_name']}")
        print(f"Total Time:          {results['timing']['total_time']}s")
        print(f"  - Prediction:      {results['timing']['predict_time']}s")
        print(f"  - Evaluation:      {results['timing']['eval_time']}s")
        print(f"Input Sentences:     {len(results['inputs'])}")
        print(f"Timestamp:           {results['timestamp']}")
        
        if args.verbose or args.debug:
            print("\nRelevance Summary:")
            pipeline.print_relevance_summary()
        
        print("\n" + "="*70)
        print("✅ Pipeline completed successfully!")
        print("="*70)
        
        if args.save_results:
            print(f"\n📁 Results saved to: {args.output_dir}/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        if args.debug or args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

