import os
import pandas as pd
import numpy as np

os.system('pip install sentence-transformers')

if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    
    categorical_columns = ['zip_code', 'formatted_work_type', 'remote_allowed', 'pay_period']
    
    model = SentenceTransformer("all-MiniLM-L6-v2") # SBERT, not case-sensitive

    input_data_path = os.path.join('/opt/ml/processing/input', 'postings.csv')

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    # create embeddings for 'title'
    print('Creating embeddings for job title')
    df["title_embeddings"] = df["title"].apply(lambda x: model.encode(x))

    # Use PCA to reduce the embeddings vector down to 2 dimensions 
    print('Reducing embeddings to 2 dimensions')
    pca = PCA(n_components=2, svd_solver='full')
    embeddings_matrix = np.vstack(df['title_embeddings'].values)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    
    df['pca_1'] = reduced_embeddings[:, 0]
    df['pca_2'] = reduced_embeddings[:, 1]
    
    # save csv to output_path
    output_path = os.path.join('/opt/ml/processing/output', 'postings.csv')
    df.to_csv(output_path, index=False)
