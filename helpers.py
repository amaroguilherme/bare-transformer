import math

def dot_product(mat1_length: int, inner_mat1: list, mat2: list[list],
                el: int = 0, i: int = 0, j: int = 0, inner_array: list = []) -> list:
    if j > mat1_length:
        return inner_array
    while i <= (len(inner_mat1) - 1):
        el += (inner_mat1[i] * mat2[i][j])
        i += 1
            
    inner_array.append(el)
    j += 1
        
    return dot_product(mat1_length=mat1_length,
                inner_mat1=inner_mat1,
                mat2=mat2,
                el=0,
                i=0,
                j=j,
                inner_array=inner_array)


def multiply_matrices(mat1: list[list], mat2: list[list]) -> list[list]:
    if not all(len(x) == len(mat1) for x in mat2):
        raise Exception("Diverging columns from matrix A x rows from matrix B")

    result: list = []
    
    for m in mat1:
        inner_array: list = dot_product(mat1_length=(len(mat1) - 1),
                           inner_mat1=m,
                           mat2=mat2,
                           inner_array=[])
        
        result.append(inner_array)
            
        
    return result


def transpose_matrix(matrix: list[list]):
    # TODO: CREATE LOGIC
    pass


def calculate_scores(q: list, k: list, d_k: int):
    transp_k = transpose_matrix(k)
    
    raw_scores = multiply_matrices(q, transp_k)
    norm_scores = raw_scores/math.sqrt(d_k)
    
    return norm_scores
            
        
def softmax(x: list) -> list:
    exponentials: list = [math.exp(score) for score in x]
    total_sum: float = sum(exponentials)
    
    prob_vector: list = [round(exp/total_sum, 3) for exp in exponentials]
    
    return prob_vector


def hyperbolic_tangent(x: float) -> float:
    return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))


def activation(x: float) -> float:
    aux_function: float = (math.sqrt(2/math.pi) * (x + 0.044715 * x**3))
    gelu: float = 0.5 * x * (1 + hyperbolic_tangent(x=aux_function))
    
    return round(gelu, 4)


def vector_sum(a: list, b: list) -> list:
    if not len(a) == len(b):
        raise Exception("Diverging length between vectors")
    
    vec = []
    
    for ax, bx in zip(a, b):
        vec.append(ax + bx)
        
    return vec


def layernorm(vec: list, weights: list, bias: list) -> list:
    learned_linear_transformation: list = vector_sum(weights, bias)
    
    avg: float = (sum(vec))/len(vec)
    standard_deviation: float = math.sqrt(sum([((v - avg)**2) for v in vec])/3)
    
    norm_vec: list = [((v - avg)/standard_deviation) * learned_linear_transformation for v in vec]
    
    return norm_vec
