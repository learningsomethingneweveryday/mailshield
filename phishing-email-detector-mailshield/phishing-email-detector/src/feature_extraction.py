"""
MailShield - Feature Extraction Module
---------------------------------------
Extracts both textual (TF-IDF) and structural URL/keyword features
from email strings for Scikit-learn classification.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "account", "password", "login", "click here",
    "bank", "confirm", "security alert", "winner", "suspended", "warning",
    "update payment", "claim", "free", "action required", "tax refund",
    "billing failure", "unauthorized", "compromised", "seed phrase",
    "direct deposit", "gift card"
]

SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly",
    "rebrand.ly", "cutt.ly", "goo.gl", "shorturl.at"
]

URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)
IP_URL_REGEX = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s]*)?', re.IGNORECASE)


def extract_url_and_keyword_features(text: str) -> Dict[str, Any]:
    """Extract structural defensive features from email text without network requests."""
    if not isinstance(text, str):
        text = ""
        
    lower_text = text.lower()
    urls = URL_REGEX.findall(text)
    num_urls = len(urls)
    
    num_http_links = sum(1 for u in urls if u.lower().startswith("http://"))
    num_https_links = sum(1 for u in urls if u.lower().startswith("https://"))
    
    ip_urls = IP_URL_REGEX.findall(text)
    num_ip_urls = len(ip_urls)
    
    num_shortened_urls = 0
    for u in urls:
        u_lower = u.lower()
        if any(domain in u_lower for domain in SHORTENER_DOMAINS):
            num_shortened_urls += 1
            
    matched_keywords = []
    num_suspicious_keywords = 0
    for kw in SUSPICIOUS_KEYWORDS:
        count = len(re.findall(r'\b' + re.escape(kw) + r'\b', lower_text))
        if count > 0:
            num_suspicious_keywords += count
            matched_keywords.append(kw)
            
    alpha_chars = [c for c in text if c.isalpha()]
    uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    
    return {
        "num_urls": num_urls,
        "num_http_links": num_http_links,
        "num_https_links": num_https_links,
        "num_ip_urls": num_ip_urls,
        "num_shortened_urls": num_shortened_urls,
        "num_suspicious_keywords": num_suspicious_keywords,
        "text_length": len(text),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "matched_keywords": matched_keywords,
        "extracted_urls": urls
    }


class URLAndKeywordExtractor(BaseEstimator, TransformerMixin):
    """Custom Scikit-learn Transformer for structural URL and keyword metrics."""
    def __init__(self):
        self.feature_names_ = [
            "num_urls", "num_http_links", "num_https_links",
            "num_ip_urls", "num_shortened_urls", "num_suspicious_keywords",
            "text_length", "uppercase_ratio"
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features_list = []
        for text in X:
            f = extract_url_and_keyword_features(text)
            features_list.append([
                f["num_urls"],
                f["num_http_links"],
                f["num_https_links"],
                f["num_ip_urls"],
                f["num_shortened_urls"],
                f["num_suspicious_keywords"],
                f["text_length"] / 500.0,
                f["uppercase_ratio"]
            ])
        return np.array(features_list, dtype=np.float32)


def get_tfidf_vectorizer(max_features: int = 1500) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )


def build_combined_feature_union() -> FeatureUnion:
    return FeatureUnion([
        ("tfidf", get_tfidf_vectorizer()),
        ("url_keyword_feats", URLAndKeywordExtractor())
    ])
