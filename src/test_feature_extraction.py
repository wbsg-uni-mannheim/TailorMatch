import json
import pandas as pd
from uncertainty_classifier import UncertaintyFeatureExtractor

def test_feature_extraction():
    # Load the data file
    results_path = "/ceph/aasteine/fine-tuning-paper/results/meta-llama/Meta-Llama-3.1-8B-Instruct/regular_train_small_all_fields/lr_0.0002/2025-05-12-12-20-23/results/wdc-fullsize/2025-06-03-09-33-08_lama3.json"
    
    df = pd.read_json(results_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get the first sample
    sample = df.iloc[0]
    top_probs = sample.get('top_probs', {})
    
    print(f"Top probs type: {type(top_probs)}")
    print(f"Top probs keys: {list(top_probs.keys()) if isinstance(top_probs, dict) else 'N/A'}")
    
    # Test feature extraction
    extractor = UncertaintyFeatureExtractor()
    
    try:
        features = extractor.extract_features(top_probs)
        print(f"\nSuccessfully extracted {len(features)} features:")
        for key, value in list(features.items())[:10]:  # Show first 10 features
            print(f"  {key}: {value}")
        
        print(f"\nKey feature values:")
        print(f"  max_yes_prob: {features.get('max_yes_prob', 'N/A')}")
        print(f"  max_no_prob: {features.get('max_no_prob', 'N/A')}")
        print(f"  confidence_ratio: {features.get('confidence_ratio', 'N/A')}")
        print(f"  num_generation_steps: {features.get('num_generation_steps', 'N/A')}")
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        print(f"Top probs structure: {top_probs}")

if __name__ == "__main__":
    test_feature_extraction() 