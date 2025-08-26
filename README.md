# Bare Transformer 

Implementation of the **Transformer architecture** from the ground up, without relying on deep learning libraries. The goal is to fully understand and recreate each component of the model step by step.  

## Current Progress  
- [x] **Attention Block** (Scaled Dot-Product Attention + Multi-Head Attention)  
- [x] Feed Forward Network  
- [ ] Positional Encoding  
- [ ] Encoder  
- [ ] Decoder  
- [ ] Full Transformer (Encoder-Decoder)  
- [ ] Training Loop (custom backpropagation)  
- [ ] Example: Machine Translation / Text Generation  

## 📖 Overview  

The Transformer architecture was introduced in *“Attention is All You Need” (Vaswani et al., 2017)* and has become the backbone of modern LLMs. It eliminates recurrence and convolutions in favor of **attention mechanisms**.  

### Key Components  
1. **Attention Mechanism**  
   - Scaled Dot-Product Attention  
   - Multi-Head Attention  
2. **Positional Encoding**  
3. **Feed Forward Layers**  
4. **Residual Connections + Layer Normalization**  
5. **Encoder-Decoder Architecture**  

This repo rebuilds these blocks step by step, ensuring full transparency of every mathematical operation.  

## Attention Block  

Currently implemented:  
- Query, Key, Value transformations  
- Scaled dot-product attention  
- Multi-head attention (splitting & concatenation)  
- Attention masking (optional for causal/decoder use)  

## Feed Forward Block  

Implemented:  
- Two-layer position-wise feed forward network
- Non-linearity using GeLU  
- Applied independently to each position   

## 📌 Next Steps  
- Add positional encodings  
- Implement the feed forward block  
- Stack encoder layers  
- Add the decoder and train a toy model  

## 📜 References  
- Vaswani et al., *Attention Is All You Need* (2017)  
- Annotated Transformer by Harvard NLP  
- The Illustrated Transformer by Jay Alammar  

