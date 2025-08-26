from helpers import activation
from typing import List

class FFN:
    
    def __init__(self,
                 x: List[List],
                 w1: List[List], w2: List[List],
                 b1: List, b2: List):
        self.x = x,
        self.w1 = w1,
        self.w2 = w2,
        self.b1 = b1,
        self.b2 = b2
        
    
    def enrich(self):
        return (self.w2 * activation((self.w1 * self.x) + self.b1)) + self.b2
