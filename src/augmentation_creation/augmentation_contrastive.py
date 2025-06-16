import numpy as np
np.random.seed(42)
import random
random.seed(42)
import pandas as pd



import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

from nltk.tokenize import word_tokenize


def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    """Return existing cluster_id or assign a new one."""
    try:
        return cluster_id_dict[identifier]
    except KeyError:
        return cluster_id_amount


def serialize_sample_lspc(sample):
    """Serialize sample for lspc dataset."""
    string = ''
    string = f"{string}[COL] brand [VAL] {' '.join(sample['brand'].split()[:5])}".strip()
    string = f"{string} [COL] title [VAL] {' '.join(sample['title'].split()[:50])}".strip()
    string = f"{string} [COL] description [VAL] {' '.join(sample['description'].split()[:100])}".strip()
    string = f"{string} [COL] specTableContent [VAL] {' '.join(sample['specTableContent'].split()[:200])}".strip()
    return string


def serialize_sample_abtbuy(sample):
    """Serialize sample for abt-buy dataset."""
    string = ''
    string = f"{string}[COL] brand [VAL] {' '.join(sample['brand'].split())}".strip()
    string = f"{string} [COL] title [VAL] {' '.join(sample['name'].split())}".strip()
    string = f"{string} [COL] price [VAL] {' '.join(str(sample['price']).split())}".strip()
    string = f"{string} [COL] description [VAL] {' '.join(sample['description'].split()[:100])}".strip()
    return string


def serialize_sample_amazongoogle(sample):
    """Serialize sample for amazon-google dataset."""
    string = ''
    string = f"{string}[COL] brand [VAL] {' '.join(sample['manufacturer'].split())}".strip()
    string = f"{string} [COL] title [VAL] {' '.join(sample['title'].split())}".strip()
    string = f"{string} [COL] price [VAL] {' '.join(str(sample['price']).split())}".strip()
    string = f"{string} [COL] description [VAL] {' '.join(sample['description'].split()[:100])}".strip()
    return string


class SplitAug:
    """Custom word-splitting augmenter: randomly splits tokens at a random position."""
    def __init__(self, stopwords=None, aug_p=0.1, tokenizer=word_tokenize):
        self.split_p = aug_p
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
        return ' '.join(new_tokens)


class Augmenter:
    """Wrapper to randomly apply one of several augmenters (or none)."""
    def __init__(self, aug: str):
        stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']
        # Define individual augmenters with 10% per-token augmentation
        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=0.1, aug_word_p=0.1)
        aug_swap = naw.RandomWordAug(action='swap', stopwords=stopwords, aug_p=0.1)
        aug_delete = naw.RandomWordAug(action='delete', stopwords=stopwords, aug_p=0.1)
        aug_crop = naw.RandomWordAug(action='crop', stopwords=stopwords, aug_p=0.1)
        aug_substitute = naw.RandomWordAug(action='substitute', stopwords=stopwords, aug_p=0.1)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=0.1)

        choice = aug.strip('-').lower()
        if choice == 'all':
            self.augs = [aug_typo, aug_swap, aug_delete, aug_crop, aug_substitute, aug_split, None]
        elif choice == 'typo':
            self.augs = [aug_typo, None]
        elif choice == 'swap':
            self.augs = [aug_swap, None]
        elif choice == 'delete':
            self.augs = [aug_delete, None]
        elif choice == 'crop':
            self.augs = [aug_crop, None]
        elif choice == 'substitute':
            self.augs = [aug_substitute, None]
        elif choice == 'split':
            self.augs = [aug_split, None]
        else:
            raise ValueError(f"Unknown augmentation type: {aug}")

    def apply_aug(self, text: str) -> str:
        """Randomly apply one augmentation or return text unchanged."""
        aug_choice = random.choice(self.augs)
        if aug_choice is None:
            return text
        return aug_choice.augment(text)


if __name__ == "__main__":
    # load dataset
    dataset = pd.read_pickle('/work/aasteine/example_selection/data/wdc/train_small/preprocessed_wdcproducts80cc20rnd000un_train_small_explanations_40.pkl.gz', compression='gzip')

    print(dataset["label"].value_counts())
    
    def create_prompt_without_explanation(product1: str, product2: str, answer: str, id: str):
        correct = 'Yes' if answer == 1 else 'No'
        prompt = f"Do the two product descriptions refer to the same real-world product? Entity 1: '{product1}'. Entity 2: '{product2}'."

        return {
            "prompt": prompt,
            "completion": correct,
            "id": id
        }

    # Create augmented versions of the dataset
    #augmentation_types = ['typo', 'swap', 'delete', 'crop', 'substitute', 'split', 'all']
    augmentation_types = ['all']
    for aug_type in augmentation_types:
        print(f"\nProcessing {aug_type} augmentation...")

        augmenter = Augmenter(aug_type)
        
        # Create augmented versions of product descriptions
        def augment_product(row, all_attributes=False):
            augmented_records = []
            try:
                if all_attributes:
                    # Augment left product
                    left_product = f"{row['brand_left']} {row['title_left']} {row['description_left']} {row['price_left']} {row['priceCurrency_left']}"
                    augmented_left = augmenter.apply_aug(left_product)[0]
                else:
                    left_product = row['title_left']
                    augmented_left = augmenter.apply_aug(left_product)[0]
                
                # Augment right product
                if all_attributes:
                    right_product = f"{row['brand_right']} {row['title_right']} {row['description_right']} {row['price_right']} {row['priceCurrency_right']}"
                    augmented_right = augmenter.apply_aug(right_product)[0]
                else:
                    right_product = row['title_right']
                    augmented_right = augmenter.apply_aug(right_product)[0]
                
                # Create three augmented versions:
                # 1. Left augmented, right original
                augmented_records.append(create_prompt_without_explanation(
                    product1=augmented_left,
                    product2=right_product,
                    answer=row["label"],
                    id= f"{row['pair_id']}_left_augmented_right_original"
                ))
                # 2. Left original, right augmented
                augmented_records.append(create_prompt_without_explanation(
                    product1=left_product,
                    product2=augmented_right,
                    answer=row["label"],
                    id= f"{row['pair_id']}_left_original_right_augmented"
                ))
                # 3. Both augmented
                augmented_records.append(create_prompt_without_explanation(
                    product1=augmented_left,
                    product2=augmented_right,
                    answer=row["label"],
                    id= f"{row['pair_id']}_both_augmented"
                ))
                
                return augmented_records
            except Exception as e:
                print(f"Error augmenting row: {str(e)}")
                left_product = row['title_left']
                right_product = row['title_right']
                # Return original if augmentation fails
                return [create_prompt_without_explanation(
                    product1=left_product,
                    product2=right_product,
                    answer=row["label"],
                    id= f"{row['pair_id']}_original"
                )]
        
        # Create original dataset
        """
        df_original = dataset.apply(lambda row: create_prompt_without_explanation(
            product1=f"{row['brand_left']} {row['title_left']} {row['description_left']} {row['price_left']} {row['priceCurrency_left']}", 
            product2=f"{row['brand_right']} {row['title_right']} {row['description_right']} {row['price_right']} {row['priceCurrency_right']}", 
            answer=row["label"]), axis=1, result_type='expand')
        """
        df_original = dataset.apply(lambda row: create_prompt_without_explanation(
            product1=f"{row['title_left']}", 
            product2=f"{row['title_right']}", 
            answer=row["label"],
            id= f"{row['pair_id']}_original"
            ), axis=1, result_type='expand')
        
        # only augment matching examples
        matching_examples = dataset[dataset["label"] == 1]
        # Apply augmentation to each row and collect all augmented records
        all_augmented_records = []
        for _, row in matching_examples.iterrows():
            augmented_records = augment_product(row)
            all_augmented_records.extend(augmented_records)
            
        # show an example of the augmented records
        print(f"Original records: {df_original.iloc[0]['prompt']}")
        print(f"Example of augmented records: {all_augmented_records[0:3]}")
        
        # Convert augmented records to DataFrame
        df_augmented = pd.DataFrame(all_augmented_records)
        
        # Combine original and augmented data
        df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
        
        # shuffle
        df_combined = df_combined.sample(frac=1).reset_index(drop=True)
        
        print(f"Label distribution: {df_combined['completion'].value_counts()}")
        
        # Save combined dataset
        output_path = f"data/wdc/train_small/augmentation/preprocessed_wdcproducts80cc20rnd000un_train_small_{aug_type}_augmentation_nplaug_matching_examples_v2.csv"
        df_combined.to_csv(output_path, index=False)
        print(f"Saved {aug_type} augmented dataset to {output_path}")