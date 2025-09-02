from data_preprocessing import DataPreprocessor


def test_build_model_vocabulary():
    preprocessor = DataPreprocessor()
    text_list = ["The cat is in the box"]
    
    preprocessor.build_vocab(text_list)
    
    assert preprocessor.vocab == {'the': 4, 'cat': 5, 'is': 6, 'in': 7, 'box': 8}
    

def test_encode_into_integer_sequence():
    preprocessor = DataPreprocessor()
    text_list = ["The cat is in the box"]
    
    preprocessor.build_vocab(text_list)
    encoded_sequence = preprocessor.encode(text_list[0])
    
    assert encoded_sequence == [4, 5, 6, 7, 4, 8]
    

def test_decode_into_text():
    preprocessor = DataPreprocessor()
    text_list = ["The cat is in the box"]
    
    preprocessor.build_vocab(text_list)
    encoded_sequence = preprocessor.encode(text_list[0])
    
    text = preprocessor.decode(encoded_sequence)
    
    assert text == text_list[0].lower().split()


def test_build_embedding_matrix():
    preprocessor = DataPreprocessor()
    text_list = ["The cat is in the box"]

    preprocessor.build_vocab(text_list)
    matrix = preprocessor.build_embedding_matrix()

    # Check dimensions
    assert len(matrix) == len(preprocessor.vocab) + len(preprocessor.special_tokens)
    assert all(len(row) == preprocessor.embedding_dim for row in matrix)

    # Check if all values are part of the interval [-lim, lim]
    lim = 1 / (preprocessor.embedding_dim ** 0.5)
    for row in matrix:
        for value in row:
            assert -lim <= value <= lim
            

def test_calculate_positional_encoding():
    preprocessor = DataPreprocessor()
    text_list = ["The cat is in the box"]
    
    preprocessor.build_vocab(text_list)
    encoded_sequence = preprocessor.encode(text_list[0])
    
    pe = preprocessor.calculate_positional_encoding(encoded_sequence)
    
    assert len(pe) == 6
    assert all(len(row) == preprocessor.embedding_dim for row in pe)