import math
import random

from helpers import multiply_matrices
from typing import List


class Attention:
    
    def __init__(self,
                 embedding_vector: List[List],
                 wq: List[List] = None,
                 wk: List[List] = None,
                 wv: List[List] = None):
        
        """
            attention_heads is defaulted as 8, as per original transformer
        """
        
        self.embedding_vector = embedding_vector
        self.dimensions = embedding_vector.dimensions
        
        self.attention_heads = 8
        self.d_k = self.dimensions/self.attention_heads
        
        self.wq = wq
        self.wk = wk
        self.wv = wv
        
        self.q = None
        self.k = None
        self.v = None
        
    
    def _get_statistic_distribution(self):
        lim = math.sqrt((6/(self.dimensions + self.d_k)))
        
        return lim
    
    
    def _initialize_dense_matrices(self):
        lim = self._get_statistic_distribution()
        
        if not self.wq and self.wk and self.wv:
            for _ in range(self.dimensions):
                self.wq = [random.choices(range(-lim, lim), k=self.d_k) for _ in range(self.dimensions)]
                self.wk = [random.choices(range(-lim, lim), k=self.d_k) for _ in range(self.dimensions)]
                self.wv = [random.choices(range(-lim, lim), k=self.d_k) for _ in range(self.dimensions)]
        
    
    def _get_input_linear_projections(self):
        if not (self.wq and self.wk and self.wv):
            self._initialize_dense_matrices()
            
        self.q = multiply_matrices(self.embedding_vector, self.wq)
        self.k = multiply_matrices(self.embedding_vector, self.wk)
        self.v = multiply_matrices(self.embedding_vector, self.wv)