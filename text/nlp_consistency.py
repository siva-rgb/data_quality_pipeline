from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

model = SentenceTransformer('all-MiniLM-L6-v2')

def detect_text_inconsistencies_columnwise(df, columns, eps=0.6, min_samples=1):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = {}

    for col in columns:
        values = df[col].astype(str).values
        embeddings = model.encode(values)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        print(f"\nColumn: {col}")
        print("Labels:", labels)
        print("Values:", list(values))

        anomalies = []
        for i, label in enumerate(labels):
            if label == -1:
                anomalies.append({
                    "row": int(i),
                    "value": values[i]
                })

        if anomalies:
            results[col] = anomalies

    return  results
