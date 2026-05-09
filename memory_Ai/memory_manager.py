import chromadb
import time
import os
class VectoryManagerMemory:
    def __init__(self, persistence_dir=None):
        if persistence_dir is None:
            persistence_dir = os.path.dirname(os.path.abspath(__file__))
        # Initialize ChromaDB client and collection for memory storage
        self.client = chromadb.PersistentClient(path=persistence_dir)
        
        # (all-Mini LM-L6-v2) is a good general-purpose embedding model, but you can choose others based on your needs
        # get or create a collection named "saiko_memory" for storing memory vectors
        self.collection = self.client.get_or_create_collection(name="saiko_memory")

    def add_memory_interaction(self, user_msg, assistant_msg):
        """Save dialogue in vector database with embeddings."""
        doc_id = f"msg_{int(time.time())}" # Unique ID based on timestamp
        memory_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        self.collection.add(documents=[memory_text], ids=[doc_id])
        print(f"Memory interaction saved with ID: {doc_id}")

    def get_relevant_context(self, current_query, top_k=3):
        """We are looking for similar past dialogues by meaning"""
        if self.collection.count() == 0:
            return "No previous memories." 
        
        # don't try to get more results than we have in memory
        limit = min(top_k, self.collection.count())

        results = self.collection.query(
            query_texts=[current_query],
            n_results=limit
        )
        if results and results["documents"] and len(results["documents"][0]) > 0:
            return "\n".join(results["documents"][0])  # Join top results into a single string
        return "NO RELEVANT MEMORY FOUND"

