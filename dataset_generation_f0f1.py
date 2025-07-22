import pandas as pd
import os
import itertools
from tqdm import tqdm
import joblib 
import json
import warnings

warnings.simplefilter("ignore", category=UserWarning)

merge_directory = "/shared/3/projects/bangzhao/prosodic_embeddings/merge/output_phones/"
output_dir = "/shared/3/projects/bangzhao/prosodic_embeddings/merge/training_data_f0f1/"

# Load clustering models (each is a tuple: (kmeans, scaler))
kmeans10, scaler10 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_10clu_f0f1.pkl")
kmeans20, scaler20 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_20clu_f0f1.pkl")
kmeans50, scaler50 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_50clu_f0f1.pkl")
kmeans100, scaler100 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_100clu_f0f1.pkl")
kmeans200, scaler200 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_200clu_f0f1.pkl")
kmeans500, scaler500 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_500clu_f0f1.pkl")
kmeans1000, scaler1000 = joblib.load("/shared/3/projects/bangzhao/prosodic_embeddings/merge/clustering/kmeans_10000epi_1000clu_f0f1.pkl")
print('done.')

files = [f for f in os.listdir(merge_directory) if os.path.isfile(os.path.join(merge_directory, f))]
print('done.')

BATCH_SIZE = 500      # Write to file every 500 processed files
MAX_LINES = 50000     # Start a new file after another 50,000 lines

buffer = []           # Temporary storage for records
total_lines = 0   # Start from 50,000 since the first part is done
file_count = 1       

# Create new output JSONL file for the second part
output_jsonl = os.path.join(output_dir, f"output_part_{file_count}.jsonl")
jsonl_file = open(output_jsonl, "w", encoding="utf-8")

files = files[0:]

# Process files
for idx, filename in enumerate(tqdm(files, desc="Processing files", unit="file")):
    file_path = os.path.join(merge_directory, filename)

    try:
        # Read CSV file
        data = pd.read_csv(file_path)
        data = data.dropna(subset=['F0semitoneFrom27.5Hz_sma3nz', 'F1frequency_sma3nz'])
        
        X = data[['F0semitoneFrom27.5Hz_sma3nz', 'F1frequency_sma3nz']]
        
        # X = data[['F0semitoneFrom27.5Hz_sma3nz', 'F1frequency_sma3nz', 'mfcc1_sma3', 'mfcc2_sma3', 
        #           'mfcc3_sma3', 'mfcc4_sma3', 'F0semitoneFrom27.5Hz_sma3nzSlope', 'F1frequency_sma3nzSlope',
        #           'mfcc1_sma3Slope', 'mfcc2_sma3Slope', 'mfcc3_sma3Slope', 'mfcc4_sma3Slope']]

        # Apply clustering
        # Scale features before prediction
        X_scaled_10 = scaler10.transform(X)
        X_scaled_20 = scaler20.transform(X)
        X_scaled_50 = scaler50.transform(X)
        X_scaled_100 = scaler100.transform(X)
        X_scaled_200 = scaler200.transform(X)
        X_scaled_500 = scaler500.transform(X)
        X_scaled_1000 = scaler1000.transform(X)
        
        # Apply clustering
        prosody_id_10 = kmeans10.predict(X_scaled_10)
        prosody_id_20 = kmeans20.predict(X_scaled_20)
        prosody_id_50 = kmeans50.predict(X_scaled_50)
        prosody_id_100 = kmeans100.predict(X_scaled_100)
        prosody_id_200 = kmeans200.predict(X_scaled_200)
        prosody_id_500 = kmeans500.predict(X_scaled_500)
        prosody_id_1000 = kmeans1000.predict(X_scaled_1000)

        # Extract phoneme column
        phonemes = data['content']

        # Create DataFrame for result
        df_result = pd.DataFrame({
            'phoneme': phonemes, 
            'prosody_id_10': prosody_id_10,
            'prosody_id_20': prosody_id_20,
            'prosody_id_50': prosody_id_50,
            'prosody_id_100': prosody_id_100,
            'prosody_id_200': prosody_id_200,
            'prosody_id_500': prosody_id_500,
            'prosody_id_1000': prosody_id_1000,
        })

        # Convert DataFrame to JSON format
        json_record = {
            "name": filename,  # Store filename
            **df_result.to_dict(orient="list")  # Convert all columns to lists
        }

        # Add record to buffer
        buffer.append(json.dumps(json_record))
        total_lines += 1

        # Write buffer to file when it reaches BATCH_SIZE
        if len(buffer) >= BATCH_SIZE:
            jsonl_file.write("\n".join(buffer) + "\n")
            buffer = []  # Clear buffer

        # If total_lines reaches MAX_LINES, start a new file
        if total_lines >= MAX_LINES:
            jsonl_file.close()  # Close current file
            break
            file_count += 1      # Increment file counter
            total_lines = 0      # Reset line counter

            # Create new JSONL file
            output_jsonl = os.path.join(output_dir, f"output_part_{file_count}.jsonl")
            jsonl_file = open(output_jsonl, "w", encoding="utf-8")
            

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Write any remaining records in buffer
if buffer:
    jsonl_file.write("\n".join(buffer) + "\n")

# Close the last file
jsonl_file.close()

print(f"Processed remaining {len(files)} files. JSONL files saved in {output_dir}.")
