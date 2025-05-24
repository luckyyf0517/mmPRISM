import os
import json
import argparse
from src.eval.metrics import translation_performance

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics from saved test results")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing JSON files with test results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save detailed metric results (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load all JSON files in the specified directory
    all_predictions = []
    all_references = []
    
    for filename in os.listdir(args.results_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.results_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                for sample_id, data in results.items():
                    prediction = data["prediction"]
                    # if '对对对对' in prediction:
                    #     # print('skip bad samples')
                    #     continue
                    prediction = ' '.join(list(prediction.replace(" ",'').replace("\n",'')))
                    reference = data["reference"]
                    reference = ' '.join(list(reference.replace(" ",'').replace("\n",'')))
                    all_predictions.append(prediction)
                    all_references.append(reference)
    
    print(f"Loaded {len(all_predictions)} samples for evaluation")
    
    # Calculate evaluation metrics
    bleu_dict, rouge_score = translation_performance(all_references, all_predictions)
    
    # Print the results
    print(bleu_dict.keys())
    print(f"BLEU 1: {bleu_dict['bleu1']:.2f}")
    print(f"BLEU 4: {bleu_dict['bleu4']:.2f}")
    print(f"ROUGE-L F1: {rouge_score:.2f}")
    
    # If an output directory is specified, save the detailed results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        metrics = {
            "bleu": bleu_dict,
            "rouge_l_f1": rouge_score,
            "sample_count": len(all_predictions)
        }
        
        metrics_file = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Detailed metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main() 