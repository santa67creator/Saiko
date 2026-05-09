import chromadb
import os
import time
def main_menu():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "memory_AI"))
    try:
        collection = client.get_collection(name="saiko_memory")
    except Exception as e:
        print("The collection was not found. Please interact with the bot first.")
        return

    while True:
        print("\n=== YOUR MEMORY EDITOR ===")
        print("1. Find memory (search by meaning/word)")
        print("2. Delete memory (by ID)")
        print("3. Add new fact manually")
        print("0. Exit")
        
        choice = input("Choose an action: ")
        
        if choice == '1':
            query = input("Enter a word or phrase to search for (e.g., 'name' or 'game'): ")
            results = collection.query(query_texts=[query], n_results=5)
            
            print("\n--- RESULTS ---")
            if not results['ids'][0]:
                print("Nothing found.")
            else:
                for i in range(len(results['ids'][0])):
                    print(f"ID: {results['ids'][0][i]}")
                    print(f"Text: {results['documents'][0][i]}")
                    print("-" * 30)
                    
        elif choice == '2':
            doc_id = input("Enter the ID of the memory you want to delete (e.g., msg_1710000000): ")
            try:
                collection.delete(ids=[doc_id])
                print(f"✅ Record {doc_id} successfully deleted!")
            except Exception as e:
                print(f"❌ Error occurred while deleting: {e}")
                
        elif choice == '3':
            new_fact = input("Enter the new fact you want to add (e.g., 'User fact: My name is SanSan'): ")
            new_id = f"manual_{int(time.time())}"
            collection.add(documents=[new_fact], ids=[new_id])
            print(f"✅ Fact added with ID: {new_id}")
            
        elif choice == '0':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_menu()