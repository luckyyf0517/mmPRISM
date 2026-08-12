import os
import json
import argparse
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from src.eval.metrics import translation_performance
from collections import defaultdict
from tqdm import tqdm

class Evaluator:
    def __init__(self, model_dir="./pretrained_models"):
        self.simcse_dir = os.path.join(model_dir, "simcse")
        self.sbert_dir = os.path.join(model_dir, "sbert")
        
        print("Loading SimCSE model...")
        self.simcse_tokenizer = AutoTokenizer.from_pretrained(self.simcse_dir)
        self.simcse_model = AutoModel.from_pretrained(self.simcse_dir).to("cuda")
        
        print("Loading SBERT model...")
        self.sbert_model = SentenceTransformer(self.sbert_dir, device="cuda")

    def batch_eval(self, all_pred, all_gt, gt_count, batch_size=32):
        len_of_pred = len(all_pred)
        all_sbert_sim = []
        all_simcse_sim = []
        
        # Process in batches
        for i in tqdm(range(0, len_of_pred, batch_size), desc="Processing batches"):
            batch_pred = all_pred[i:i + batch_size]
            batch_gt = all_gt[i:i + batch_size]
            
            with torch.no_grad():
                # Get embeddings for current batch
                sbert_embeddings = self.sbert_model.encode(batch_pred + batch_gt, show_progress_bar=False, device="cuda")
                inputs = self.simcse_tokenizer(batch_pred + batch_gt, padding=True, truncation=True, return_tensors="pt").to("cuda")
                simcse_embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

            batch_pred_sbert_embed = sbert_embeddings[:len(batch_pred)]
            batch_pred_simcse_embed = simcse_embeddings[:len(batch_pred)]
            batch_gt_sbert_embed = sbert_embeddings[len(batch_pred):]
            batch_gt_simcse_embed = simcse_embeddings[len(batch_pred):]

            # Calculate similarities for current batch
            for j in range(len(batch_pred)):
                # Compare with corresponding ground truth
                sbert_similarity = util.cos_sim(batch_pred_sbert_embed[j], batch_gt_sbert_embed[j])[0][0].item()
                simcse_similarity = 1 - cosine(batch_pred_simcse_embed[j].cpu().detach().numpy(), 
                                            batch_gt_simcse_embed[j].cpu().detach().numpy())
                all_sbert_sim.append(sbert_similarity)
                all_simcse_sim.append(simcse_similarity)
            
            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
            
        return all_sbert_sim, all_simcse_sim

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics from saved test results")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing JSON files with test results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save detailed metric results (optional)")
    parser.add_argument("--model_dir", type=str, default="./pretrained_models",
                        help="Directory containing pretrained models")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing semantic similarity metrics")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load all JSON files in the specified directory
    all_predictions = []
    all_references = []
    
    print("Loading evaluation data...")
    for filename in tqdm(os.listdir(args.results_dir), desc="Loading files"):
        if filename.endswith('.json'):
            file_path = os.path.join(args.results_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                for sample_id, data in results.items():
                    prediction = data["prediction"].replace(" ", "").replace("\n", "")
                    reference = data["reference"].replace(" ", "").replace("\n", "")
                    all_predictions.append(prediction)
                    all_references.append(reference)
    
    print(f"Loaded {len(all_predictions)} samples for evaluation")
    
    # Initialize evaluator for semantic similarity metrics
    evaluator = Evaluator(model_dir=args.model_dir)
    
    # Calculate traditional metrics using translation_performance
    print("\nCalculating traditional metrics...")
    bleu_dict, rouge_score = translation_performance(
        [' '.join(list(ref)) for ref in all_references],
        [' '.join(list(pred)) for pred in all_predictions]
    )
    
    # Calculate semantic similarity metrics
    print("\nCalculating semantic similarity metrics...")
    gt_count = [1] * len(all_predictions)
    all_sbert_sim, all_simcse_sim = evaluator.batch_eval(
        all_predictions, all_references, gt_count, batch_size=args.batch_size
    )
    
    # Print the results
    print("\nTraditional Metrics:")
    print(f"BLEU 1: {bleu_dict['bleu1']:.4f}")
    print(f"BLEU 2: {bleu_dict['bleu2']:.4f}")
    print(f"BLEU 3: {bleu_dict['bleu3']:.4f}")
    print(f"BLEU 4: {bleu_dict['bleu4']:.4f}")
    print(f"ROUGE-L F1: {rouge_score:.4f}")
    
    print("\nSemantic Similarity Metrics:")
    print(f"SBERT Similarity: {np.mean(all_sbert_sim):.4f}")
    print(f"SimCSE Similarity: {np.mean(all_simcse_sim):.4f}")
    
    # If an output directory is specified, save the detailed results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        metrics = {
            "traditional_metrics": {
                "bleu": bleu_dict,
                "rouge_l_f1": rouge_score
            },
            "semantic_similarity": {
                "sbert": float(np.mean(all_sbert_sim)),
                "simcse": float(np.mean(all_simcse_sim))
            },
            "sample_count": len(all_predictions)
        }
        
        metrics_file = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nDetailed metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main() 