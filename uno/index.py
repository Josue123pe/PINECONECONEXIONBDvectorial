import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Cargar variables del .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

index_name = "chatcitobot"

# Crear índice solo si no existe
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"   # Región válida en plan FREE
            }
        }
    )
    print(f"Índice '{index_name}' creado correctamente.")
else:
    print(f"Índice '{index_name}' ya existe. Continuando...")

# Conectarse al índice
index = pc.Index(index_name)
print("Conectado al índice:", index_name)

# Crear un vector de prueba (1536 dimensiones)
vector = [0.1, 0.2, 0.3, 0.4] * 384

# Subir vector
index.upsert([("vec_1", vector)])
print("Vector insertado correctamente.")

# Consulta de similitud
query = [0.1, 0.2, 0.3, 0.4] * 384
result = index.query(vector=query, top_k=3)

print(result)



