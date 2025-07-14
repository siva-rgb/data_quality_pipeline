from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def detect_column_inconsistencies(df, catagorical_column,use_prompt_contxt=None, threshold=0.8):
    """
    Detect inconsistencies in specified columns of a DataFrame using sentence embeddings.

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: list of column names to check for inconsistencies.
    - threshold: cosine similarity threshold to consider values as consistent.

    Returns:
    - A dictionary with column names as keys and lists of inconsistent values as values.
    """
    missing_value= {'','NA', 'N/A', 'NaN', 'nan', 'None', 'null', None}
    results= []
    report= []

    for col in catagorical_column:
        all_values= df[col].astype(str).unique().tolist()
        clean_values= [val for val in all_values if val.strip() not in missing_value]

        if not clean_values:
            continue
    # contex vector refrence
        # context_entry= use_prompt_contxt.get(col)
        # if isinstance(context_entry, list):
        #     reference_text= context_entry
        # elif isinstance(context_entry, str):
        #     reference_text= [context_entry]
        # else:
        #     reference_text= clean_values

        # reference_embedings= model.encode(reference_text)
        # reference_vector= np.mean(reference_embedings, axis=0)

    #embede column value
        embedings= model.encode(clean_values)
        mean_embedings= np.mean(embedings, axis=0)
        similarities= cosine_similarity([mean_embedings], embedings)[0]

        threshold= 0.70

        for i, sim in enumerate(similarities):
            if sim >= threshold:
                results.append({
                'column': col,
                'value': clean_values[i],
                'similarity': round(sim, 2),
                'issue': 'Potential inconsistency or anomaly'
            }   )
    
        for val in all_values:
            if val.strip() in missing_value:
                results.append({
                    'column': col,
                    'value': val,
                    'similarity': None,
                    'issue': 'Missing value'
                })
        # Fin outliers report
        outliers=[(val, sim) for val,sim in zip(clean_values, similarities) if sim< threshold]

        report.append(
            {
                "column":col,
                "unique_values": len(clean_values),
                "outliers_detected": len(outliers),
                "outlier_pct": round((len(outliers)/len(clean_values))*100,2),
                "avg_similarity": round(np.mean(similarities),2),
                "Min_similarity": round(np.min(similarities),2),
                "Max_similarity": round(np.max(similarities),2),
                "Top_Outliers": sorted(outliers, key=lambda x: x[1], reverse=True)[:5]
            }
        )

        # Vsiualize the embeddings using t-SNE
        if len(clean_values)<3:
            continue

        labels= ['Outliers' if sim< threshold else 'Normal' for sim in similarities]
        tsne= TSNE(n_components=2,perplexity=5, random_state=42)
        reduced= tsne.fit_transform(embedings)

        plt.figure(figsize=(10,6))
        for lable_type in set(labels):
            label_indices= [i for i, label in enumerate(labels) if label==lable_type]
            plt.scatter(
                [reduced[i][0] for i in label_indices],
                [reduced[i][1] for i in label_indices],
                label=lable_type,
                s=100,
                alpha=0.7
            )
        for i, txt in enumerate(clean_values):
            plt.annotate(txt, (reduced[i][0], reduced[i][1]), fontsize=9)
        plt.title(f'TSNE plot for {col}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return results, report


def detect_text_anomalies_dbscan(df, catagorical_column, eps=0.6, min_samples=1):
    """
    Detect text inconsistencies in specified columns of a DataFrame using DBSCAN clustering.

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: list of column names to check for inconsistencies.
    - eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    - A dictionary with column names as keys and lists of detected anomalies as values.
    """
    result= []
    missing_value= {'','NA', 'N/A', 'NaN', 'nan', 'None', 'null', None}
    for col in catagorical_column:
        all_values= df[col].astype(str).apply(str.strip).unique().tolist()
        clean_values= [val for val in all_values if val.strip() not in missing_value]
        embedings= model.encode(clean_values)
        db= DBSCAN(eps=0.5, min_samples=5, metric='cosine', algorithm='brute')
        cluster= db.fit_predict(embedings)


        

        for i, lable in enumerate(cluster):
            result.append({
                'column': col,
                'value': clean_values[i],
                'cluster': lable,
                'issue': 'Outlier' if lable == -1 else 'Normal'
            })

    # result_df= pd.DataFrame(result)
    return result
