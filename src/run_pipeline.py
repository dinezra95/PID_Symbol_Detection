"""Main entry point for P&ID symbol detection system."""

import argparse
from pathlib import Path
from typing import Optional

from pipeline.stage1_class_agnostic_train import Stage1ClassAgnosticPipeline
from pipeline.stage2_few_shot_train import Stage2FewShotPipeline
from pipeline.stage1_class_agnostic_inference import Stage1InferencePipeline
from pipeline.stage2_few_shot_inference import Stage2FewShotInferencePipeline
from pipeline.stage3_pipe_detection import Stage3PipeDetectionPipeline
from pipeline.evaluation import EvaluationPipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="P&ID Symbol Detection System")
    
    # Common arguments
    parser.add_argument("--config", type=str, default="src/pipeline/configs/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Stage 1 command
    stage1_parser = subparsers.add_parser("stage1", help="Run stage 1 pipeline")
    stage1_parser.add_argument("--prepare_data", action="store_true",
                              help="Prepare data for stage 1")
    stage1_parser.add_argument("--train_model", action="store_true",
                              help="Train stage 1 model")
    
    # Stage 2 command
    stage2_parser = subparsers.add_parser("stage2", help="Run stage 2 pipeline")
    stage2_parser.add_argument("--prepare_data", action="store_true",
                              help="Prepare data for stage 2")
    stage2_parser.add_argument("--train_model", action="store_true",
                              help="Train stage 2 model")
    
    # stage 1 inference command
    stage1_inference_parser = subparsers.add_parser("stage1_inference", help="Run stage 1 inference")
    stage1_inference_parser.add_argument("--input", type=str, default=None,
                              help="Input directory with images (overrides config)")
    stage1_inference_parser.add_argument("--output", type=str, default=None,
                              help="Output directory for results (overrides config)")
    stage1_inference_parser.add_argument("--model", type=str, default=None,
                              help="Path to model weights (overrides config)")

    # stage 2 inference command
    stage2_inference_parser = subparsers.add_parser("stage2_inference", help="Run stage 2 inference")
    stage2_inference_parser.add_argument("--input", type=str, default=None,
                              help="Input directory with images (overrides config)")
    stage2_inference_parser.add_argument("--labels", type=str, default=None,
                              help="Stage 1 labels directory (overrides config)")
    stage2_inference_parser.add_argument("--output", type=str, default=None,
                              help="Output directory for results (overrides config)")

    # Stage 3 pipe detection command
    stage3_parser = subparsers.add_parser("stage3_pipes", help="Run stage 3 pipe detection")
    stage3_parser.add_argument("--input", type=str, default=None,
                              help="Input directory with P&ID images")
    stage3_parser.add_argument("--labels", type=str, default=None,
                              help="Directory with stage1 YOLO label .txt files")
    stage3_parser.add_argument("--output", type=str, default=None,
                              help="Output directory for pipe detection results")

    # Evaluation command
    evaluation_parser = subparsers.add_parser("evaluation", help="Run evaluation")
    evaluation_parser.add_argument("--evaluate_stage1", action="store_true",
                              help="Evaluate stage 1 model")
    evaluation_parser.add_argument("--evaluate_stage2", action="store_true",
                              help="Evaluate stage 2 model")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if args.command == "stage1":
        pipeline = Stage1ClassAgnosticPipeline(config_path=args.config)
        if args.prepare_data:
            pipeline.prepare_data()
        elif args.train_model:
            pipeline.train_model()
        else:
            pipeline.run()
            
    elif args.command == "stage2":
        pipeline = Stage2FewShotPipeline(config_path=args.config)
        if args.prepare_data:
            pipeline.prepare_symbol_crops()
            pipeline.prepare_few_shot_data()
        elif args.train_model:
            pipeline.train_model()
        else:
            pipeline.run()

    elif args.command == "stage1_inference":
        pipeline = Stage1InferencePipeline(config_path=args.config)
        pipeline.run(input_dir=args.input, output_dir=args.output, model_path=args.model)

    elif args.command == "stage2_inference":
        pipeline = Stage2FewShotInferencePipeline(config_path=args.config)
        pipeline.run(input_dir=args.input, labels_dir=args.labels, output_dir=args.output)
        
    elif args.command == "stage3_pipes":
        pipeline = Stage3PipeDetectionPipeline(config_path=args.config)
        pipeline.run(input_dir=args.input, labels_dir=args.labels, output_dir=args.output)

    elif args.command == "evaluation":
        # Assuming the EvaluationPipeline is implemented in a similar way to the others
        pipeline = EvaluationPipeline(config_path=args.config)
        if args.evaluate_stage1:
            pipeline.compute_stage1_metrics()
        elif args.evaluate_stage2:
            pipeline.compute_stage2_metrics()
        else:
            pipeline.run()
   
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main() 