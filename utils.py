import os
import hashlib

def get_metadata_file_path(identifier: str) -> str:
    """
    Get the file path for metadata based on an identifier (e.g., file path or query hash).

    :param identifier: A unique identifier (document path or query).
    :return: Full path to the metadata file.
    """
    base_dir = "/home/renyang/jadhav/rag-fyp/legalbenchrag/legalbenchrag/legalbenchrag/generated_metadata"
    os.makedirs(base_dir, exist_ok=True)
    file_name = hashlib.sha256(identifier.encode()).hexdigest()[:16] + ".json"
    print(file_name)
    # return os.path.join(base_dir, file_name)

if __name__ == "__main__":
    get_metadata_file_path("/home/renyang/jadhav/LegalBench-RAG/corpus/privacy_qa/Wordscapes.txt")
