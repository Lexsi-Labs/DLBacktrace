#!/usr/bin/env python3
"""
Example usage of the Multi-Modal DL-Backtrace Pipeline

This script demonstrates how to use the pipeline with text classification, 
image classification, and text generation models.
"""

from pipeline import DLBacktracePipeline
from model_registry import register_custom_model, ModelRegistry
from config import PipelineConfig
import numpy as np


def example_text_classification():
    """Example of text classification with sentiment analysis"""
    print("="*60)
    print("EXAMPLE 1: Text Classification (Sentiment Analysis)")
    print("="*60)
    
    # Create pipeline for text classification
    pipeline = DLBacktracePipeline.create_simple(
        model_name="bert-base",
        device="cpu",
        verbose=True,
        save_results=True,
        output_dir="text_classification_results"
    )
    
    # Sample texts and labels
    texts = [
        "This movie is absolutely fantastic!",
        "I really hate this boring film.",
        "The weather is nice today."
    ]
    labels = ["positive", "negative", "neutral"]
    
    # Run text classification analysis
    results = pipeline.run_text_classification(texts, labels)
    
    print(f"✅ Text classification completed!")
    print(f"   Processed {len(texts)} texts")
    print(f"   Model: {results['model_name']}")
    print(f"   Total time: {results['timing']['total_time']}s")


def example_image_classification():
    """Example of image classification"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Image Classification")
    print("="*60)
    
    # Create pipeline for image classification
    pipeline = DLBacktracePipeline.create_simple(
        model_name="resnet",
        device="cpu",
        verbose=True,
        save_results=True,
        output_dir="image_classification_results"
    )
    
    # Create sample images (random tensors for demo)
    # In practice, you would load actual images
    sample_images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    ]
    labels = ["cat", "dog"]
    
    try:
        # Run image classification analysis
        results = pipeline.run_image_classification(sample_images, labels)
        
        print(f"✅ Image classification completed!")
        print(f"   Processed {len(sample_images)} images")
        print(f"   Model: {results['model_name']}")
        print(f"   Total time: {results['timing']['total_time']}s")
        
    except Exception as e:
        print(f"⚠️ Image classification failed: {e}")
        print("💡 This might require additional dependencies like PIL/Pillow")


def example_text_generation():
    """Example of text generation with relevance analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Text Generation with Relevance")
    print("="*60)
    
    # Create pipeline for text generation
    config = PipelineConfig(
        model_name="llama3.2-1b",
        device="cpu",
        verbose=True,
        
        # Generation parameters
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_beams=1,
        return_relevance=True,
        
        # Output configuration
        save_results=True,
        output_dir="generation_results"
    )
    
    pipeline = DLBacktracePipeline(config)
    
    # Sample prompts
    prompts = [
        "The future of artificial intelligence",
        "Climate change is a serious issue that",
        "In the world of technology,"
    ]
    
    try:
        # Run text generation analysis
        results = pipeline.run_text_generation(prompts)
        
        print(f"✅ Text generation completed!")
        print(f"   Processed {len(prompts)} prompts")
        print(f"   Generated texts: {len(results.get('generated_texts', []))}")
        print(f"   Model: {results['model_name']}")
        print(f"   Total time: {results['timing']['total_time']}s")
        
        # Show sample outputs
        if results.get('generated_texts'):
            for i, (prompt, generated) in enumerate(zip(prompts, results['generated_texts'])):
                print(f"\n   Prompt {i+1}: {prompt}")
                print(f"   Generated: {generated}")
        
    except Exception as e:
        print(f"⚠️ Text generation failed: {e}")
        print("💡 Make sure the model supports generation")


def example_advanced_generation():
    """Example of advanced text generation with custom parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Text Generation")
    print("="*60)
    
    # Create pipeline with advanced generation settings
    pipeline = DLBacktracePipeline.create_simple(
        model_name="qwen3-0.6b",
        device="cpu",
        verbose=False
    )
    
    # Single prompt with custom generation parameters
    prompt = "Once upon a time in a magical forest"
    
    try:
        # Run generation with custom parameters
        results = pipeline.run_text_generation(
            [prompt],
            max_new_tokens=50,
            temperature=1.2,
            top_k=40,
            top_p=0.95,
            num_beams=2,
            num_return_sequences=2,
            return_scores=True,
            return_relevance=False  # Skip relevance for faster generation
        )
        
        print(f"✅ Advanced generation completed!")
        print(f"   Prompt: {prompt}")
        
        if results.get('generated_texts'):
            for i, text in enumerate(results['generated_texts']):
                print(f"   Generated {i+1}: {text}")
        
        if results.get('generation_scores'):
            print(f"   Generation scores: {results['generation_scores']}")
        
    except Exception as e:
        print(f"⚠️ Advanced generation failed: {e}")


def example_custom_models():
    """Example of registering and using custom models"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Model Registration")
    print("="*60)
    
    # Register custom models for different tasks
    
    # Custom text classification model
    register_custom_model(
        model_name="my-roberta-sentiment",
        base_model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_type="text_classification",
        modality="text",
        description="Custom RoBERTa model for sentiment analysis"
    )
    
    # Custom generation model
    register_custom_model(
        model_name="my-qwen-medium",
        base_model_path="Qwen/Qwen2.5-3B",
        model_type="generation",
        modality="text",
        description="Custom Qwen 3B model"
    )
    
    print("✅ Custom models registered!")
    
    # Use custom text classification model
    try:
        pipeline = DLBacktracePipeline.create_simple(
            model_name="my-roberta-sentiment",
            device="cpu",
            save_results=False
        )
        
        results = pipeline.run_text_classification([
            "I love this new feature!",
            "This is terrible."
        ])
        
        print(f"✅ Custom text classification completed!")
        
    except Exception as e:
        print(f"⚠️ Custom model test failed: {e}")


def example_batch_processing():
    """Example of batch processing multiple inputs"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Processing")
    print("="*60)
    
    # Create pipeline for batch text classification
    pipeline = DLBacktracePipeline.create_simple(
        model_name="bert-base",
        device="cpu",
        batch_size=4,  # Process 4 samples at once
        verbose=False,
        save_results=False
    )
    
    # Large batch of texts
    texts = [
        "This product is amazing!",
        "Worst purchase ever.",
        "It's okay, nothing special.",
        "Absolutely love it!",
        "Not worth the money.",
        "Pretty good quality.",
        "Fantastic service!",
        "Very disappointed."
    ]
    
    try:
        # Process all texts in batch
        results = pipeline.run_text_classification(texts)
        
        print(f"✅ Batch processing completed!")
        print(f"   Processed {len(texts)} texts")
        print(f"   Batch size: {pipeline.config.batch_size}")
        print(f"   Total time: {results['timing']['total_time']}s")
        print(f"   Time per sample: {results['timing']['total_time']/len(texts):.3f}s")
        
    except Exception as e:
        print(f"⚠️ Batch processing failed: {e}")


def list_available_models():
    """List all available models by category"""
    print("\n" + "="*60)
    print("AVAILABLE MODELS BY CATEGORY")
    print("="*60)
    
    models = ModelRegistry.list_models()
    
    # Group by model type
    categories = {
        "Text Classification": [],
        "Image Classification": [],
        "Text Generation": []
    }
    
    for name, description in models.items():
        model_info = ModelRegistry.get_model_info(name)
        if model_info.model_type == "text_classification":
            categories["Text Classification"].append((name, description))
        elif model_info.model_type == "image_classification":
            categories["Image Classification"].append((name, description))
        elif model_info.model_type == "generation":
            categories["Text Generation"].append((name, description))
    
    for category, model_list in categories.items():
        if model_list:
            print(f"\n📂 {category}:")
            for name, desc in model_list:
                print(f"   📦 {name}: {desc}")


if __name__ == "__main__":
    print("🚀 Multi-Modal DL-Backtrace Pipeline Examples")
    print("Supporting Text Classification, Image Classification, and Text Generation")
    
    # List available models
    list_available_models()
    
    try:
        # Run examples for all three task types
        example_text_classification()
        example_image_classification()
        example_text_generation()
        example_advanced_generation()
        example_custom_models()
        example_batch_processing()
        
        print("\n" + "="*60)
        print("✅ All multi-modal examples completed successfully!")
        print("="*60)
        print("\n💡 Key Features Demonstrated:")
        print("   • Text Classification with BERT/RoBERTa/ALBERT/DistilBERT/XLNet/ELECTRA")
        print("   • Image Classification with VGG/ResNet/ViT/DenseNet/EfficientNet/MobileNet")
        print("   • Text Generation with Llama3.2/Qwen3")
        print("   • Custom model registration")
        print("   • Batch processing")
        print("   • Relevance analysis for all tasks")
        print("   • Flexible configuration options")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install torch transformers torchvision pillow numpy")