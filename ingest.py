from vectorstore import VectorStore

def split_text(text, chunk_size=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


if __name__ == "__main__":
    
    # company data 
    text = """
    Our company provides AI solutions to enterprise clients.
    We specialize in machine learning, cloud computing, and data analytics.
    Employees can access internal knowledge using this system.
    We use Python, AWS, and modern AI tools to build scalable systems.
    """

    chunks = split_text(text)

    store = VectorStore()
    store.add_text(chunks)

    print(" Data ingested successfully")