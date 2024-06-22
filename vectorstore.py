import time
from sentence_transformers import SentenceTransformer
from typing import *
import numpy as np 
from enum import Enum
from functools import wraps

from documents import document

class similarity_metric(Enum):
    COSINE = 0
    EUCLIDEAN = 1

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper


class TinyVectorStore:
    """
    A tiny vector store implementation built with numpy.
    """
    def __init__(
        self, 
        docs: List[str],
        embedding_model: SentenceTransformer = None,
        metric: similarity_metric = similarity_metric.COSINE
    ) -> None:
        self.docs = np.array(docs)
        self.embedding_model = None or embedding_model
        self.metric = metric

        # main store
        self.store: np.ndarray = None
        self.similarity_metric = metric
        self.set_sim_func()

    def set_dist_metric(self, metric: similarity_metric):
        assert isinstance(metric, similarity_metric), "Invalid metric"
        self.similarity_metric = metric
        self.sim_func = sefl.set_sim_func()

    def set_sim_func(self):
        if self.similarity_metric == similarity_metric.COSINE:
            self.sim_func = self.cosine_sim
        elif self.similarity_metric == similarity_metric.EUCLIDEAN:
            self.sim_func = self.euclidean_sim
        else:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")

    @classmethod
    def from_docs(
        cls, 
        docs: List[str], 
        embedding_model: SentenceTransformer = None, 
        metric = similarity_metric.EUCLIDEAN
    ) -> "TinyVectorStore":

        """
        Initializes the process of building a TinyVectorStore from a list of documents.
        """
        store = cls(docs, embedding_model=embedding_model, metric=metric)
        print(f"Using similarity metric: {metric}")
        return store.build_store()

    def build_store(self):
        """
        Embeds the documents into vectors/tensors and builds the store.
        """
        if self.embedding_model is not None:
            self.store = self.embedding_model.encode(self.docs)
        else:
            raise ValueError("Embedding model not set")
        return self


    @timer
    def search(self, query:str, k:int=5):
        """
        Searches for the top k most similar documents to the query. Returns the cosine similarity 
        scores.

        For euclidian distance: Lower scores are better (vector coordinates are closer to each other - hence more similar)
        For cosine similarity: Higher scores are better (vectors are more similar)
        """
        assert self.embedding_model is not None, "Embedding model not set"
        assert k > 0, "k must be greater than 0"

        query_embedding = self.embedding_model.encode(query)

        assert query_embedding.ndim == 1, "Query should be a 1 dimensional vector"

        return self.get_top_k_scores(query_embedding, k)

    def euclidean_dist(self, query_embedding: np.ndarray):
        """
        Calculates Euclidian distance as a similarity metric for the query embedding and the store.

        Uses the following formula:
        
        sqrt(sum((x1 - x2)^2))

        where x1 and x2 are the coordinates of the query and store embeddings respectively.
        """
        assert query_embedding.ndim == 1, "Query embedding should only be 1D"
        assert query_embedding.shape[0] == self.store.shape[1], "Query embedding dimension mismatch"

        dist:np.ndarray = np.sqrt(np.sum((self.store - query_embedding)**2, axis=1))

        return dist

    def cosine_sim(self, query_embedding: np.ndarray):
        """
        Calculates cosine similarity as a similarity metric for the query embedding and the store.

        Uses the following formula:
        (x1.x2) / (||x1|| * ||x2||)

        where x1 and x2 are the store and query embeddings respectively.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
    
        assert query_embedding.shape[1] == self.store.shape[1], "Query embedding dimension mismatch"

        norm_query = np.linalg.norm(query_embedding, axis=1)
        norm_store = np.linalg.norm(self.store, axis=1)
        similarity:np.ndarray = np.dot(self.store, query_embedding.T).T / (norm_query * norm_store)
    
        return similarity.flatten()  # Ensure the output is always 1D
    
    def get_top_k_scores(self, query_embedding, k):
        """
        Get the top k most similar documents to the query embedding.

        Also returns the corresponding similarity score for the document based on the query embedding.
        """
        reverse = True if self.similarity_metric == similarity_metric.COSINE else False

        sim_scores = self.sim_func(query_embedding)
        sorted_sim_scores_indices = np.argsort(sim_scores)
        top_k_indices = sorted_sim_scores_indices[::-1][:k] if reverse else sorted_sim_scores_indices[:k]

        top_k_docs = [self.docs[i] for i in top_k_indices]
        top_k_scores = sim_scores[top_k_indices]

        return zip(top_k_docs, top_k_scores) #zip to create pairs of the docs and sim scores

    
    def __repr__(self):
        return f"TinyVectorStore(embedding_model = {self.embedding_model})"

    
# --------- Example Usage -------------

# Example

print("loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
docs = document.split("\n")

print(f"Loaded {len(docs)} documents")
vectorstore = TinyVectorStore.from_docs(docs, embedding_model=model, metric=similarity_metric.COSINE)
print(f"Building TinyVectorStore for querying with {len(docs)} documents")

# Search for the most similar documents to a query
query = "What is the quote?" 
results, execution_time = vectorstore.search(query, k=2)


print(f"Most similar documents = {results[0]}")
print(f"Similarity score = {results[1]} using '{vectorstore.similarity_metric.name}' metric")