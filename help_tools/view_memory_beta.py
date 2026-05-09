import chromadb
import os
def view_all_memory():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "memory_AI"))
    try:
        # Пытаемся получить нашу коллекцию
        collection = client.get_collection(name="saiko_memory")
    except Exception as e:
        print("The database is empty or the collection was not found.")
        return

    # Получаем ВСЕ записи (documents) и их айдишники (ids)
    data = collection.get()
    
    total_records = len(data['ids'])
    print(f"=== YOUR VECTOR MEMORY ===")
    print(f"Total records: {total_records}\n")
    
    if total_records == 0:
        print("Memory is currently empty.")
        return

    # Выводим каждую запись красиво
    for i in range(total_records):
        doc_id = data['ids'][i]
        text = data['documents'][i]
        
        print(f"[{i+1}] ID: {doc_id}")
        print(f"Content:\n{text}")
        print("-" * 50)

if __name__ == "__main__":
    view_all_memory()