
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Guardrail(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_calibrated = False
        self.threshold = None
        
    @abstractmethod
    def process(self, text: str, context: Dict) -> Dict:
        """Returns: {'output': str, 'triggered': bool, 'score': float, 'metadata': dict}"""
        pass
    
    def calibrate_to_fpr(self, benign_texts: List[str], target_fpr: float = 0.05, tolerance: float = 0.01):
        """Calibrate threshold to achieve target false positive rate"""
        scores = []
        for text in benign_texts:
            result = self.process(text, {})
            scores.append(result['score'])
        
        # Binary search for threshold that gives target FPR
        scores = sorted(scores, reverse=True)
        n = len(scores)
        target_idx = int(target_fpr * n)
        
        if target_idx < n:
            self.threshold = scores[target_idx]
        else:
            self.threshold = min(scores) - 0.1
            
        self.is_calibrated = True

class KeywordGuardrail(Guardrail):
    """DFA-based guardrail using regex patterns"""
    
    def __init__(self, patterns: List[str], name: str = "keyword"):
        super().__init__(name)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        
    def process(self, text: str, context: Dict) -> Dict:
        matches = []
        max_score = 0.0
        
        for pattern in self.patterns:
            pattern_matches = pattern.findall(text)
            if pattern_matches:
                matches.extend(pattern_matches)
                # Score based on number and length of matches
                score = len(pattern_matches) + sum(len(m) for m in pattern_matches) / 100
                max_score = max(max_score, score)
        
        triggered = (self.threshold is not None and max_score > self.threshold) or (self.threshold is None and len(matches) > 0)
        
        output = text
        if triggered:
            for pattern in self.patterns:
                output = pattern.sub("[REDACTED]", output)
                
        return {
            'output': output,
            'triggered': triggered,
            'score': max_score,
            'metadata': {'matches': matches, 'pattern_count': len(matches)}
        }

class EmbeddingGuardrail(Guardrail):
    """Statistical guardrail using TF-IDF similarity"""
    
    def __init__(self, harmful_examples: List[str], name: str = "embedding"):
        super().__init__(name)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.harmful_vectors = self.vectorizer.fit_transform(harmful_examples)
        
    def process(self, text: str, context: Dict) -> Dict:
        try:
            text_vector = self.vectorizer.transform([text])
            similarities = cosine_similarity(text_vector, self.harmful_vectors)
            max_sim = float(np.max(similarities))
        except:
            max_sim = 0.0
            
        triggered = (self.threshold is not None and max_sim > self.threshold) or (self.threshold is None and max_sim > 0.5)
        
        output = text if not triggered else f"[Content filtered - similarity: {max_sim:.3f}]"
        
        return {
            'output': output,
            'triggered': triggered,
            'score': max_sim,
            'metadata': {'max_similarity': max_sim}
        }

class ComposedGuardrail(Guardrail):
    """Serial composition of two guardrails"""
    
    def __init__(self, first: Guardrail, second: Guardrail):
        super().__init__(f"{first.name}+{second.name}")
        self.first = first
        self.second = second
        
    def process(self, text: str, context: Dict) -> Dict:
        # Apply first guardrail
        result1 = self.first.process(text, context)
        
        # Apply second to output of first
        result2 = self.second.process(result1['output'], context)
        
        return {
            'output': result2['output'],
            'triggered': result1['triggered'] or result2['triggered'],
            'score': max(result1['score'], result2['score']),  # Simple aggregation
            'metadata': {
                'first': result1,
                'second': result2,
                'both_triggered': result1['triggered'] and result2['triggered']
            }
        }
