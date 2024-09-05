import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd

def calculate_cosine_similarity_matrix(arrays):
    n = len(arrays)
    cosine_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            try:
                flat_i = np.array(arrays[i], dtype=float).flatten()
                flat_j = np.array(arrays[j], dtype=float).flatten()

                cosine_matrix[i, j] = 1 - cosine(flat_i, flat_j)

            except:
                cosine_matrix[i, j] = np.nan

    return cosine_matrix

# Take average of off-diagonal cosine similarity results
def calculate_quality_score(matrix):
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return np.nanmean(off_diag)

def save_results_to_csv(cosine_matrix, quality_score, filename='cosine_similarity_results.csv'):
    df_cosine = pd.DataFrame(cosine_matrix, columns=[f't{i+1}' for i in range(len(cosine_matrix))])
    df_cosine['t'] = [f't{i+1}' for i in range(len(cosine_matrix))]

    cols = ['t'] + [col for col in df_cosine.columns if col != 't']
    df_cosine = df_cosine[cols]

    df_score = pd.DataFrame({
        'Metric': ['Quality Score'],
        'Value': [quality_score]
    })

    with open(filename, 'w', newline='') as f:
        df_cosine.to_csv(f, index=False)
        f.write('\n')
        df_score.to_csv(f, index=False)

def analyze_cosine_similarity(pvd, output_path, save=True):
    arrays = [pvd.mip[i] for i in range(4)]
    
    cosine_matrix = calculate_cosine_similarity_matrix(arrays)
    quality_score = calculate_quality_score(cosine_matrix)
    
    if save:
        save_results_to_csv(cosine_matrix, quality_score, filename=f"{output_path}mip_cosine_similarity.csv")
    
    return cosine_matrix, quality_score

# Create summary of matched segment and between-timepoint cosine similarity for MIPs
# probably should turn this into a dataframe/.csv
def data_summary(files, datasets, sessions, summary_file = "data_summary.txt"):
    # Summary Report
    for ii, file in enumerate(files):
        results_path = 'pvd_analysis'
        dataset = datasets[ii]
        session = sessions[ii]
        output_path = f"{results_path}/{dataset}/{session}/"

        segment_csv = pd.read_csv(f"{output_path}segment_change.csv")
        quality_csv = pd.read_csv(f"{output_path}mip_cosine_similarity.csv")
        csv_length = segment_csv.shape[1]-3  # Subtract 3 for index and core segment
        quality_score = quality_csv.loc[5,'t1']  # Get avg. cosine similarity score from csv

        mode = 'w' if ii == 0 else 'a'
        with open(summary_file, mode) as file:
            file.write(f"{dataset} - {session} -- Segments: {csv_length}  Quality: {float(quality_score):.2f}\n")