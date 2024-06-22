# tiny_vecstore

tiny_vecstore is a small vector store implemented using just numpy.

# Why I made this

On twitter, I see a lot of stuff about RAG, vectorDBs, vector stores, etc.

In those threads, people throw around a lot of jargon that I don't understand, and they make it seem like its quite complicated. However, after a bit of research, I've found that vector stores are surprisingly simple to implement, at least simpler than you think.

# What it is

tiny_vecstore is a simple ~180 line implementation of a vector store that is pretty fast even on my M2 macbook air.

It's not meant to be a full-fledged production grade solution, but rather a simple example that is easy to understand and implement.

Here is it at work with a draft of my writing:

#### Input
```
document = """
Written by Benyamin Damircheli, May 28th 2024
My thoughts on when and what to share about your work and life.

“Man is not what he thinks he is, he is what he hides.” -André Malraux

Before you get the wrong impression, this essay is not about why you should live a life of secrecy, nor am I encouraging you to lie. This is about why you should delay the gratification that you get from sharing your work and life with others.

(More blog post here but too long to include ...)
"""

query = "What is the quote?" 
results, execution_time = vectorstore.search(query, k=2)

print(f"Most similar documents = {results[0]}")
print(f"Similarity score = {results[1]} using '{vectorstore.similarity_metric.name}' metric")
```

#### Output

```
search took 0.08317184448242188 seconds
Most similar documents = “Man is not what he thinks he is, he is what he hides.” -André Malraux
Similarity scores = 0.4935154616832733 using 'COSINE' metric
```

This is awesome because no where in the stored document have I written the word "quote", yet it is able to find and return the quote, because it semantically understood what I meant! The implications of this are vast and exciting. One example use case I can think of is for knowledge and "second brain" apps.


# What I learned making this
- Gained a greater intuition of how embedding and semantic search works.
- Learned how to implement cosine similarity  and euclidean distance as similarity metrics for embeddings.
- Understood the importance of normalizing embeddings for cosine similarity.
- Learned/reinforced my understanding of the linear algebra behind embeddings and semantic search (cosine similarity, euclidean distance, normalization, dotproduct, transposition, etc)
- Practiced writing - what I think is - clean, understandable and maintainable code.
- Reinforced my OOP and numpy skills.
