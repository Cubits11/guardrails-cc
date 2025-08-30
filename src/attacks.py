import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class AttackResult:
    prompt: str
    success_score: float  # Higher = more successful at distinguishing worlds
    metadata: Dict

class Attacker(ABC):
    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
    @abstractmethod
    def generate_prompts(self, n: int, context: Dict) -> List[str]:
        pass

class RandomAttacker(Attacker):
    """Baseline: random prompts from a pool"""
    
    def __init__(self, prompt_pool: List[str], seed: int = 42):
        super().__init__("random", seed)
        self.prompt_pool = prompt_pool
        
    def generate_prompts(self, n: int, context: Dict) -> List[str]:
        return [self.rng.choice(self.prompt_pool) for _ in range(n)]

class GeneticAttacker(Attacker):
    """Simple genetic algorithm - mutate successful prompts"""
    
    def __init__(self, base_prompts: List[str], population_size: int = 50, mutation_rate: float = 0.3, seed: int = 42):
        super().__init__("genetic", seed)
        self.population = base_prompts[:population_size]
        self.mutation_rate = mutation_rate
        self.fitness_history = []
        
    def mutate_prompt(self, prompt: str) -> str:
        """Simple mutations: word substitution, insertion, deletion"""
        words = prompt.split()
        if not words:
            return prompt
            
        mutation_type = self.rng.choice(['substitute', 'insert', 'delete'])
        
        if mutation_type == 'substitute' and words:
            idx = self.rng.randint(0, len(words) - 1)
            # Simple synonym substitution (you could use a real synonym dict)
            synonyms = {
                'secret': ['confidential', 'private', 'hidden'],
                'information': ['data', 'details', 'facts'],
                'tell': ['reveal', 'show', 'explain'],
                'password': ['code', 'key', 'token']
            }
            if words[idx].lower() in synonyms:
                words[idx] = self.rng.choice(synonyms[words[idx].lower()])
                
        elif mutation_type == 'insert':
            idx = self.rng.randint(0, len(words))
            new_word = self.rng.choice(['please', 'actually', 'really', 'maybe', 'just'])
            words.insert(idx, new_word)
            
        elif mutation_type == 'delete' and len(words) > 2:
            idx = self.rng.randint(0, len(words) - 1)
            words.pop(idx)
            
        return ' '.join(words)
    
    def generate_prompts(self, n: int, context: Dict) -> List[str]:
        # Use fitness history to guide selection if available
        if self.fitness_history and len(self.fitness_history) == len(self.population):
            # Select top performers for mutation
            fitness_array = np.array(self.fitness_history)
            top_indices = np.argsort(fitness_array)[-n//2:]
            selected = [self.population[i] for i in top_indices]
        else:
            selected = self.rng.sample(self.population, min(n//2, len(self.population)))
            
        prompts = []
        for prompt in selected:
            prompts.append(prompt)  # Keep original
            if self.rng.random() < self.mutation_rate:
                prompts.append(self.mutate_prompt(prompt))  # Add mutant
                
        # Fill remaining with random from population
        while len(prompts) < n:
            prompts.append(self.rng.choice(self.population))
            
        return prompts[:n]
    
    def update_fitness(self, prompts: List[str], fitness_scores: List[float]):
        """Update population based on fitness"""
        # Simple: replace worst performers with best prompts
        if len(prompts) == len(fitness_scores):
            combined = list(zip(prompts, fitness_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 80% of population, replace bottom 20% with top performers
            keep_n = int(0.8 * len(self.population))
            self.population = self.population[:keep_n] + [p for p, f in combined[:len(self.population) - keep_n]]
            self.fitness_history = [f for p, f in combined[:len(self.population)]]
          
