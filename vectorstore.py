from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.text_chunks = []
        self.vectors = None

        #  default data 
        self.load_default_data()

    def load_default_data(self):
        text = """
        Our company provides AI solutions to enterprise clients.
        We specialize in machine learning, cloud computing, and data analytics.
        Employees can access internal knowledge using this system.
        """

        self.text_chunks = [text]
        self.vectors = self.vectorizer.fit_transform(self.text_chunks)

    def search(self, query, k=2):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)
        top_indices = similarities.argsort()[0][-k:][::-1]
        return [self.text_chunks[i] for i in top_indices]