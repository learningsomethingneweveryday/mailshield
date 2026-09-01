"""
MailShield - Preprocessing Module
---------------------------------
Handles data loading, cleaning, duplicate removal, text normalization,
and label encoding for the Phishing Email Detection project.
"""

import re
import string
import pandas as pd

# Default English stop words list for beginner-friendly standalone preprocessing
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the emails dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f" Dataset successfully loaded from '{filepath}'")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Dataset file not found at '{filepath}'.")


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect and handle missing values in the dataset."""
    missing_count = df.isnull().sum()
    print(" Missing values per column:")
    for col, count in missing_count.items():
        print(f"   - {col}: {count}")
        
    if missing_count.sum() > 0:
        initial_len = len(df)
        df = df.dropna(subset=['text', 'label']).copy()
        print(f"   🧹 Dropped {initial_len - len(df)} rows containing missing values.")
    else:
        print("   ✅ No missing values detected.")
        
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and remove duplicate emails from the dataset."""
    duplicate_count = df.duplicated(subset=['text']).sum()
    print(f" Duplicate entries: {duplicate_count}")
    
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=['text']).copy()
        print(f"   🧹 Removed duplicates. New dataset size: {len(df)} samples.")
    else:
        print("   ✅ No duplicates found.")
        
    return df


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Preprocess raw email text: lowercase, strip punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in STOP_WORDS]
        text = ' '.join(words)
        
    return text


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical labels into binary numerical values (phishing: 1, safe: 0)."""
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    label_map = {'phishing': 1, 'safe': 0, 'spam': 1, 'legitimate': 0, 'ham': 0}
    df['label_num'] = df['label'].map(label_map)
    return df


def preprocess_dataframe(df: pd.DataFrame, remove_stopwords: bool = False) -> pd.DataFrame:
    """Full pipeline execution: Clean text, check missing, remove duplicates, encode labels."""
    df = check_missing_values(df)
    df = remove_duplicates(df)
    df = encode_labels(df)
    df['cleaned_text'] = df['text'].apply(lambda t: clean_text(t, remove_stopwords=remove_stopwords))
    return df
