import pytest

from helpers import multiply_matrices, softmax, activation


def test_matrix_multiplication_basic_matrix():
    matrix_a = [[1, 2, 3], [4, 5, 6]]
    matrix_b = [[7, 8], [9, 10], [11, 12]]
    
    result = multiply_matrices(matrix_a, matrix_b)
    
    assert result == [[58, 64], [139, 154]]
    

def test_matrix_multiplications_square_matrices():
    matrix_a = [[1, 2], [4, 5]]
    matrix_b = [[7, 8], [9, 10]]
    
    result = multiply_matrices(matrix_a, matrix_b)
    
    assert result == [[25, 28], [73, 82]]
    

def test_matrix_multiplications_invalid_multiplication():
    matrix_a = [[1, 2], [4, 5], [6, 7]]
    matrix_b = [[7, 8], [9, 10]]
    
    with pytest.raises(Exception) as exc:
        multiply_matrices(matrix_a, matrix_b)
    
    assert str(exc.value) == "Diverging columns from matrix A x rows from matrix B"
    
    
def test_softmax():
    z = [2.0, 1.0, 0.1]
    
    result = softmax(z)
    
    assert result == [0.659, 0.242, 0.099]
    

def test_activation_function_for_random_x_value():
    x = 0.4694
    
    result = activation(x)
    
    assert result == 0.3195