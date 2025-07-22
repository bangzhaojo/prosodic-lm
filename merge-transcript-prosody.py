import os
import pandas as pd
import json
import sys
import re

# prosody
import time
import numpy as np
import audb
import audiofile
import opensmile

import requests
import subprocess
import warnings
from sklearn.linear_model import LinearRegression
import multiprocessing
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import urllib.parse


def standardize_name(name):
    return re.sub("[^\w\d.]","", name)


# Function to read the nth line and load it as a JSON object
def read_nth_line_as_json(file_path, n):
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == n - 1:  # Since line numbering starts from 0, we use n - 1
                    # Parse the line as a JSON object
                    return json.loads(line)
        return f"Line {n} does not exist in the file."
    except json.JSONDecodeError as json_err:
        return f"Error decoding JSON: {json_err}"
    except Exception as e:
        return f"Error reading the file: {e}"
    

def download_and_convert(json_object, name, download_directory):
    """Download MP3, save transcript, and convert MP3 to WAV."""
    
    mp3_url = json_object['enclosure']
    
    try:
        # Download MP3
        mp3_filename = os.path.join(download_directory, f"{name}.mp3")
        response = requests.get(mp3_url, stream=True)
        if response.status_code == 200:
            with open(mp3_filename, 'wb') as mp3_file:
                for chunk in response.iter_content(chunk_size=8192):
                    mp3_file.write(chunk)
            # print(f"Downloaded MP3: {mp3_filename}")
        else:
            # print(f"Failed to download MP3 {mp3_url}, status code: {response.status_code}")
            return False

        # Convert MP3 to WAV
        wav_filename = os.path.join(download_directory, name + '.wav')
        command = ['ffmpeg', '-y', '-i', mp3_filename, wav_filename]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(mp3_filename)
        # print(f"Converted MP3 to WAV and removed original MP3: {mp3_filename}")

        return True

    except Exception as e:
        # print(f"Failed to process {name}: {e}")
        return False
    
    
def extract_prosody(wav_path, out_path):
    signal, sampling_rate = audiofile.read(
        wav_path,
        always_2d=True,
    )

    #note increasing num_workers doesn't seem to help much here 
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors, 
        num_workers=1, 
        multiprocessing=True
    )

    df = smile.process_signal(
        signal,
        sampling_rate
    )
    # df.to_csv(out_path + "LowLevel.csv")
    return df


def addMicroseconds(inStr): 
    if "." not in inStr: 
        return inStr + ".000000"  # Add microseconds if they are missing
    else: 
        return inStr

    
def getSlope(inList): 
    if len(inList) < 2:  # If there are fewer than 2 points, return 0
        return 0
    
    x = list(np.arange(0, (len(inList)*.02)-.01, .02))  # Create time points at intervals of 0.02 seconds
    x = np.array(x).reshape(-1, 1)
    y = np.array(inList).reshape(-1, 1)
    
    # Fit a linear regression line and return the slope (rate of change)
    lr = LinearRegression(fit_intercept=True).fit(x, y)
    coef = lr.coef_.item()
    return coef


def merge_prosody_transcript(prosodydf, transcriptdf, name, file_index, output_directory):
    try:
        print(f"Processing: {name}")
        toKeep = ["start", "end", 'F0semitoneFrom27.5Hz_sma3nz', 'F1frequency_sma3nz', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3','mfcc4_sma3']
        # prosodyDf = pd.read_csv(prosody_file, usecols=toKeep)
        prosodyDf = prosodydf.reset_index()[toKeep]

        # Apply the microsecond fix to the 'start' and 'end' columns
        prosodyDf["start"] = prosodyDf["start"].astype(str)
        prosodyDf["end"] = prosodyDf["end"].astype(str)
        prosodyDf["start"] = prosodyDf["start"].apply(addMicroseconds)
        prosodyDf["end"] = prosodyDf["end"].apply(addMicroseconds)

        # Extract just the time (ignoring the date) and convert it to datetime format
        prosodyDf["start"] = prosodyDf["start"].apply(lambda x: x.split(" ")[2])
        prosodyDf["start"] = pd.to_datetime(prosodyDf["start"], format="%H:%M:%S.%f")
        prosodyDf["end"] = prosodyDf["end"].apply(lambda x: x.split(" ")[2])
        prosodyDf["end"] = pd.to_datetime(prosodyDf["end"], format="%H:%M:%S.%f")

        # Convert the 'start' and 'end' times into seconds relative to the first start time
        prosodyDf["end"] = (prosodyDf["end"] - prosodyDf.loc[0, "start"]).dt.total_seconds()
        prosodyDf["start"] = (prosodyDf["start"] - prosodyDf.loc[0, "start"]).dt.total_seconds()

        # Keep every second row (removing overlapping time chunks to avoid redundancy)
        prosodyDf = prosodyDf[prosodyDf.index % 2 == 0].reset_index(drop=True)

        # Read the transcript file
        toKeep2 = ["start", "end", "content"]
        # transcriptDF = pd.read_csv(transcript_file, usecols=['Begin', 'End', 'Label'])
        transcriptDf = transcriptdf[transcriptdf['Type'] == 'phones']
        transcriptDf = transcriptDf.rename(columns={'Begin': 'start', 'End': 'end', 'Label': 'content'})

        # Reset the index to clean the DataFrame
        transcriptDf = transcriptDf.reset_index()

        # Display the first few rows of the transcript DataFrame
        transcriptDf.head(4)

        # Convert the prosody DataFrame into a list for easier processing
        prosList = prosodyDf.values.tolist()

        # Initialize lists to store average and individual prosody values
        allProsAvgs = []
        allProsVals = []

        # Initialize the prosody index for looping through prosody data
        prosIndex = 0
        prosStart = prosList[prosIndex][0]  # Start time of the first prosody chunk
        prosEnd = prosList[prosIndex][1]    # End time of the first prosody chunk
        prosMid = (prosStart + prosEnd) / 2.0  # Midpoint of the prosody chunk
        prosVals = prosList[prosIndex][2:]  # Prosody feature values for this chunk

        # Loop through each token (subtitle segment) in the transcript
        for tokIndex, tokStart, tokEnd in transcriptDf[["index", "start", "end"]].values.tolist(): 

            # Initialize an empty list to store prosody values for the current token
            currProsVals = [[] for i in range(len(prosVals))]

            # Add prosody values as long as the midpoint of the prosody chunk is within the current token's time range
            while prosMid < tokEnd:

                # For each prosody value, append it to the corresponding list
                for i, prosVal in enumerate(prosVals): 
                    currProsVals[i].append(prosVal)

                prosIndex += 1  # Move to the next prosody chunk

                # Break the loop if we've processed all the prosody data
                if prosIndex >= len(prosList): 
                    break

                # Update prosody start, end, and mid times for the new chunk
                prosStart = prosList[prosIndex][0]
                prosEnd = prosList[prosIndex][1]
                prosMid = (prosStart + prosEnd) / 2.0
                prosVals = prosList[prosIndex][2:]

            # Add the prosody values for the current token to the list
            allProsVals.append(currProsVals)


        # Convert the list of prosody values into a DataFrame
        prosodyGrouped = pd.DataFrame(allProsVals, columns=list(prosodyDf.columns)[2:])

        # Define a function to calculate the slope of the regression line for a list of values
        lr = LinearRegression(fit_intercept=True)

        # Suppress warnings for empty lists during mean/median calculations and apply the functions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prosodyAvgd = prosodyGrouped.applymap(np.mean)  # Calculate mean prosody values
            prosodyMedianed = prosodyGrouped.applymap(np.median)  # Calculate median prosody values
            prosodySlopes = prosodyGrouped.applymap(getSlope)  # Calculate slope (rate of change)

        # Add the average prosody values to the transcript DataFrame
        transcriptDf[list(prosodyDf.columns)[2:]] = prosodyAvgd

        # Add the slope values (rate of change) to the transcript DataFrame
        transcriptDf[[item + "Slope" for item in list(prosodyDf.columns)[2:]]] = prosodySlopes

        # Occasionally, transcript tokens may have zero time duration, leading to missing prosody values
        # We use forward fill to replace these missing values with the previous row's values
        transcriptDf = transcriptDf.fillna(method="ffill")
        
        # Save the resulting DataFrame
        output_path = os.path.join(output_directory, f"{file_index}-{name}.csv")
        transcriptDf.to_csv(output_path, index=False)
        print(f"Saved merged file: {file_index}-{output_path}")
        
    except Exception as e:
        print(f"Error merging files for {file_index}-{name}: {e}")
        

def process_file(i, filename, transcript_path, metadata_path, download_directory, output_directory):
    # Check if the filename is a CSV file
    if not filename.endswith(".csv"):
        return

    # Get the metadata
    line = int(filename.split('-')[0]) + 1
    metadata = read_nth_line_as_json(metadata_path, line)
    
    # Get the standardized name
    new_name = standardize_name(metadata['enclosure'])
    
    # Download MP3 from URL and convert to WAV
    wav_file_path = os.path.join(download_directory, new_name + '.wav')
    download_and_convert(metadata, new_name, download_directory)
    
    # Get prosody
    prosodydf = extract_prosody(wav_file_path, os.path.join(download_directory, new_name))
    
    # Read transcript
    file_path = os.path.join(transcript_path, filename)
    transcriptdf = pd.read_csv(file_path)
    
    # Merge prosody and transcript
    merge_prosody_transcript(prosodydf, transcriptdf, new_name, i, output_directory)
    
    # Remove the WAV file after merging
    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)
                

def main(transcript_path, metadata_path, download_directory, output_directory, numbers_list, start_line, n_processes):
    # Get all files in the directory and sort them
    all_files = [f for f in os.listdir(transcript_path) if f.endswith('.csv')]
    
    # Filter files based on the indices in numbers_list
    filtered_files = [(i, all_files[i]) for i in numbers_list if i >= start_line and i < len(all_files)]

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {
            executor.submit(
                process_file, index, filename, transcript_path, metadata_path, download_directory, output_directory
            ): filename for index, filename in filtered_files
        }

        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {filename}: {e}, skipping file.")


            
def find_missing_numbers(transcript_path):
    # Get all starting numbers from filenames in the directory
    numbers_in_directory = set()
    
    # List all files in the directory with a progress bar
    files = [f for f in os.listdir(transcript_path) if f.endswith(".csv")]
    
    for filename in files:
        try:
            # Extract the starting number from each filename
            number = int(filename.split('-')[0])
            numbers_in_directory.add(number)
        except ValueError:
            # If the filename does not start with a number, skip it
            continue

    # Define the full range of numbers we want to check
    full_range = set(range(453639))

    # Find missing numbers in the directory
    missing_numbers = sorted(full_range - numbers_in_directory)
    return missing_numbers

                
if __name__ == "__main__":
    
    metadata_path = "/shared/3/projects/benlitterer/podcastData/processed/mayJune/mayJuneDataRoles.jsonl"
    download_directory = "/shared/3/projects/bangzhao/prosodic_embeddings/merge/temp_phones/"
    output_directory = "/shared/3/projects/bangzhao/prosodic_embeddings/merge/output_phones/"
    transcript_path = '/shared/3/projects/bangzhao/prosodic_embeddings/mfa/output/'
    
    parser = argparse.ArgumentParser(description="Process JSON objects in parallel.")
    parser.add_argument('--start_line', type=int, default=0, help="The line to start processing from.")
    parser.add_argument('--num_processes', type=int, default=12, help="The number of processes to run in parallel.")
    args = parser.parse_args()
    
    numbers_list = find_missing_numbers(output_directory)
    print("Missing files:" + str(len(numbers_list)))
    
    main(transcript_path, metadata_path, download_directory, output_directory, numbers_list,
         start_line=args.start_line, n_processes=args.num_processes) 