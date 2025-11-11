# Embeddings - BPE















# Postional Encoding - Read it in depth here:
[The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

RNNs -> take input sequentially 
    They have the understanding of the importance of sequence of the input that is given to them
    Transformers ditched it. Why?
        It might capture longer dependencies in a sentence which can further lead to massive speed up 

But we need to have the sense of the order of the sequence in which words occur.

Basic thinking -> assign words some numerical value between [0,1] 0-> first words while 1 -> last word
Problem : we cannot figure out how many words are present within a specific range

So what do we want ideally?
1. The encoding should be unique for each time step
2. Distance should be consistent between two time steps 

What is proposed:
 1. Not a single number but a d-dim vector  
 2. So for each input token the embedding token (E) is added to the positonal encoding (PE) and then it is passed to the first layer of the transformer
    $X = E + PE $
3. Sinusoidal Postional Encoding Formula:

$$
\text{PE}_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

> small i -> division by a smaller number -> large arguement -> oscillates faster -- Important for self attention as it is able to understand the relative distances between the two adjacent words in a sentence


# Self Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$
