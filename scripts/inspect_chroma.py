from langchain_chroma import Chroma
from rag.config import CHROMA_DIR, get_embedding_model


def main() -> None:
    vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_model(),
    )

    count = vs._collection.count()
    print("Numero di vettori in Chroma:", count)

    sample = vs.get(limit=5)
    print("Primi 5 ids:", sample.get("ids", []))

    for doc in sample.get("documents", []):
        print("----")
        print(doc[:400])


if __name__ == "__main__":
    main()

