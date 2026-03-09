from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


class VectorStore:

    def __init__(self):
        self.texts = []
        self.index = None

    def build_index(self, texts):

        self.texts = texts

        embeddings = model.encode(texts)

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(embeddings))


    def search(self, query, top_k=3):

        if self.index is None:
            return []

        query_embedding = model.encode([query])

        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []

        for i in indices[0]:
            if i < len(self.texts):
                results.append(self.texts[i])

        return results