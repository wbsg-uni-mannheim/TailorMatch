import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
from typing import List, Dict
import random
import numpy as np
import nltk
import torch
import os
import traceback
from nltk.tokenize import word_tokenize

def setup_nltk_data():
    """Setup NLTK data with proper error handling"""
    print("Setting up NLTK data...")
    
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # List of required NLTK data
    required_data = [
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('tokenizers/punkt', 'punkt')
    ]
    
    # Download each required resource
    for path, package in required_data:
        try:
            nltk.data.find(path)
            print(f"Found {package}")
        except LookupError:
            print(f"Downloading {package}...")
            try:
                nltk.download(package, download_dir=nltk_data_dir)
                print(f"Successfully downloaded {package}")
            except Exception as e:
                print(f"Error downloading {package}: {str(e)}")
                raise

# Setup NLTK data
setup_nltk_data()

# Now import WordNet after ensuring it's available
from nltk.corpus import wordnet

# Custom word-splitting augmenter for 'ralph' option
class SplitAug:
    def __init__(self, split_p=0.1, tokenizer=word_tokenize):
        self.split_p = split_p
        self.tokenizer = tokenizer

    def augment(self, text: str) -> str:
        tokens = self.tokenizer(text)
        new_tokens = []
        for token in tokens:
            if len(token) > 1 and random.random() < self.split_p:
                idx = random.randint(1, len(token) - 1)
                new_tokens.extend([token[:idx], token[idx:]])
            else:
                new_tokens.append(token)
        return " ".join(new_tokens)

class DataAugmenter:
    def __init__(self, seed: int = 42):
        """Initialize the data augmenter with a seed for reproducibility"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        print("DataAugmenter initialized successfully")
        
    def get_base_augmenters(self) -> List[Dict]:
        """Get base level augmenters with minimal changes"""
        print("Creating base augmenters...")
        try:
            augmenters = [
                {
                    'name': 'random_substitute',
                    'augmenter': naw.RandomWordAug(
                        action="substitute",
                        aug_p=0.1,
                        aug_min=1,
                        aug_max=3
                    )
                }
            ]
            print("Base augmenters created successfully")
            return augmenters
        except Exception as e:
            print(f"Error creating base augmenters: {str(e)}")
            raise
    
    def get_medium_augmenters(self) -> List[Dict]:
        """Get medium intensity augmenters with moderate changes"""
        return [
            {
                'name': 'synonym_replacement',
                'augmenter': naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=0.2,  # 20% of words will be augmented
                    aug_min=2,  # Minimum 2 words to augment
                    aug_max=5,  # Maximum 5 words to augment
                    stopwords=None
                )
            },
            {
                'name': 'random_swap',
                'augmenter': naw.RandomWordAug(
                    action="swap",
                    aug_p=0.2,
                    aug_min=2,
                    aug_max=5
                )
            },
            {
                'name': 'back_translation',
                'augmenter': naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de',
                    to_model_name='facebook/wmt19-de-en',
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            }
        ]
    
    def get_aggressive_augmenters(self) -> List[Dict]:
        """Get aggressive augmenters with significant changes"""
        return [
            {
                'name': 'synonym_replacement',
                'augmenter': naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=0.3,  # 30% of words will be augmented
                    aug_min=3,  # Minimum 3 words to augment
                    aug_max=7,  # Maximum 7 words to augment
                    stopwords=None
                )
            },
            {
                'name': 'random_swap',
                'augmenter': naw.RandomWordAug(
                    action="swap",
                    aug_p=0.3,
                    aug_min=3,
                    aug_max=7
                )
            },
            {
                'name': 'back_translation',
                'augmenter': naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de',
                    to_model_name='facebook/wmt19-de-en',
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            },
            {
                'name': 'contextual_word_emb',
                'augmenter': naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',
                    action="substitute",
                    aug_p=0.2,
                    aug_min=2,
                    aug_max=5
                )
            }
        ]
    
    def get_ralph_augmenters(self) -> List[Dict]:
        """Get 'ralph' style augmenters: 6 types at 10% per word/char."""
        return [
            {'name': 'typo', 'augmenter': nac.KeyboardAug(aug_char_p=0.1)},
            {'name': 'word_swap', 'augmenter': naw.RandomWordAug(action='swap', aug_p=0.1)},
            {'name': 'word_delete', 'augmenter': naw.RandomWordAug(action='delete', aug_p=0.1)},
            {'name': 'span_delete', 'augmenter': naw.RandomWordAug(action='delete', aug_p=0.1, aug_min=2, aug_max=5)},
            {'name': 'synonym_replace', 'augmenter': naw.SynonymAug(aug_src='wordnet', aug_p=0.1, aug_min=1, aug_max=1, stopwords=None)},
            {'name': 'word_split', 'augmenter': SplitAug(split_p=0.1)}
        ]
    
    def augment_text(self, text: str, augmenters: List[Dict]) -> str:
        """Apply augmentation to a single text using the provided augmenters"""
        print(f"Augmenting text: {text}")
        augmented_text = text
        for aug_config in augmenters:
            try:
                print(f"Applying {aug_config['name']}...")
                augmented_text = aug_config['augmenter'].augment(augmented_text)
                print(f"Result after {aug_config['name']}: {augmented_text}")
            except Exception as e:
                print(f"Error in {aug_config['name']} ({type(e).__name__}): {e}")
                traceback.print_exc()
                continue
        return augmented_text
    
    def augment_dataset(self, texts: List[str], intensity: str = 'base') -> List[str]:
        """
        Augment a list of texts with specified intensity level
        
        Args:
            texts: List of input texts
            intensity: One of 'base', 'medium', 'aggressive', or 'ralph'
            
        Returns:
            List of augmented texts
        """
        print(f"Starting augmentation with intensity: {intensity}")
        if intensity == 'ralph':
            ralph_augs = self.get_ralph_augmenters()
            augmented_texts = []
            for text in texts:
                choices = ralph_augs + [{'name': 'identity', 'augmenter': None}]
                choice = random.choice(choices)
                if choice['augmenter'] is None:
                    augmented_texts.append(text)
                else:
                    augmented_texts.append(self.augment_text(text, [choice]))
            return augmented_texts
        elif intensity == 'base':
            augmenters = self.get_base_augmenters()
        elif intensity == 'medium':
            augmenters = self.get_medium_augmenters()
        elif intensity == 'aggressive':
            augmenters = self.get_aggressive_augmenters()
        else:
            raise ValueError("Intensity must be one of: 'base', 'medium', 'aggressive', or 'ralph'")
        
        augmented_texts = []
        for text in texts:
            try:
                augmented_text = self.augment_text(text, augmenters)
                augmented_texts.append(augmented_text)
            except Exception as e:
                print(f"Error augmenting text: {str(e)}")
                augmented_texts.append(text)  # Return original text if augmentation fails
        
        return augmented_texts

def main():
    # Example usage
    print("Initializing augmenter...")
    try:
        augmenter = DataAugmenter(seed=42)
        
        # Example texts
        texts = [
            "The product description is clear and concise.",
            "This item has excellent build quality and durability.",
            "The specifications match exactly what was advertised."
        ]
        
        # Test base augmentation
        print("\nTesting base augmentation:")
        base_augmented = augmenter.augment_dataset(texts, intensity='ralph')
        for orig, aug in zip(texts, base_augmented):
            print(f"Original: {orig}")
            print(f"Augmented: {aug}\n")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 