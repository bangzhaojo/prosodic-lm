import json
import os
import re
import requests
import shutil
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count
import concurrent.futures
import time
import argparse
import urllib.parse


metadata_path = "/shared/3/projects/benlitterer/podcastData/processed/mayJune/mayJuneDataClean.jsonl"
download_path = "/shared/3/projects/bangzhao/prosodic_embeddings/mfa/cache"
output_path = "/shared/3/projects/bangzhao/prosodic_embeddings/mfa/output"
command_history_file = "/home/bangzhao/Documents/MFA/command_history.yaml"

# Total rows: 1,124,058


def clean_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Remove content between square brackets and parentheses (including the brackets and parentheses themselves)
    cleaned_text = re.sub(r'\[.*?\]', '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text, flags=re.IGNORECASE)
    # Remove the '>>' mark
    cleaned_text = re.sub(r'>>', '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove extra spaces left after removing patterns
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Replace multiple spaces with a single space
    cleaned_text = cleaned_text.strip()  # Remove leading and trailing whitespace

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)


def clean_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                clean_text_file(file_path)
                # print(f"Processed {file_path}")
    

def download_mp3(json_object, i, download_directory):
    """Download MP3 file from the URL specified in the JSON object."""
    mp3_url = json_object['enclosure']

    response = requests.get(mp3_url, stream=True)
    if response.status_code == 200:
        mp3_filename = os.path.join(download_directory, f"{i}.mp3")
        with open(mp3_filename, 'wb') as mp3_file:
            for chunk in response.iter_content(chunk_size=8192):
                mp3_file.write(chunk)
        # print(f"Downloaded {mp3_filename}")
        return True
    else:
        # print(f"Failed to download {mp3_url}, status code: {response.status_code}")
        pass
    return False

        
def download_transcript(json_object, i, download_directory):
    """Write the transcript from the JSON object to a text file in the download directory."""
    transcript = json_object["transcript"]
    transcript_filename = os.path.join(download_directory, f"{i}.txt")
    with open(transcript_filename, 'w', encoding='utf-8') as transcript_file:
        transcript_file.write(transcript)
    # print(f"Transcript saved as {transcript_filename}")
    
    
def convert_single_mp3(mp3_path):
    try:
        wav_path = os.path.splitext(mp3_path)[0] + '.wav'
        command = ['ffmpeg', '-y', '-i', mp3_path, wav_path]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(mp3_path)
        print(f"Converted and removed: {mp3_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {mp3_path}: {e}")

def convert_mp3_to_wav(directory):
    mp3_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mp3")]
    
    # Use multiprocessing Pool with 14 processes to convert files in parallel
    with Pool(processes=14) as pool:
        pool.map(convert_single_mp3, mp3_files)
                

def run_mfa_align(input_path, acoustic_model, dictionary, output_path, beam, retry_beam, num_jobs, timeout_duration="40m"):
    # Generate a unique command history file path based on the process ID
    
    cache_file = os.path.basename(input_path)
    temp_directory = f'/shared/3/projects/bangzhao/prosodic_embeddings/mfa/temp/{cache_file}'
    os.makedirs(temp_directory, exist_ok=True)
    
    unique_id = os.getpid()
#     command_history_file = f"/shared/3/projects/bangzhao/prosodic_embeddings/mfa/temp/mfa_command_history_{unique_id}.yaml"

#     # Set the MFA_HISTORY environment variable to point to the unique file
#     env = os.environ.copy()
#     env['MFA_HISTORY'] = command_history_file

    
    # Construct the MFA align command as a single string
    command = (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate mfa && "
        f"mfa align --clean --output_format csv --final_clean --quiet "
        f"--temporary_directory {temp_directory} {input_path} {acoustic_model} {dictionary} "
        f"{output_path} --beam {beam} --retry_beam {retry_beam} --num_jobs {num_jobs} --single_speaker"
    )

    # Include the timeout command
    full_command = f"timeout {timeout_duration} bash -c '{command}'"

    # Print the command to verify
    print("Running command:", full_command)

    # Execute the command in the shell with the modified environment
    try:
        # Run the command using the shell
        subprocess.run(full_command, shell=True, check=True, executable='/bin/bash')#, env=env)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        
        if e.returncode == 124:
            print("Command timed out.")
        else:
            print(f"Failed to execute command: {e}")

    # Clean up the command history file after the command execution
    if os.path.exists(command_history_file):
        os.remove(command_history_file)
        print(f"Removed command history file: {command_history_file}")

    # Remove files or directories under /home/bangzhao/Documents/MFA/input_path

    if os.path.exists(temp_directory):
        shutil.rmtree(temp_directory)
        print(f"Removed directory at /home: {cache_file}")


def sanitize_filename(url):
    # Decode the URL-encoded characters to handle %20, etc.
    url = urllib.parse.unquote(url)
    
    # Replace illegal characters with underscores or remove them
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '_', url)
    
    # Remove any trailing periods or spaces (illegal in Windows filenames)
    sanitized = sanitized.strip().rstrip('.')

    # Optionally, truncate the filename if it's too long (common limit is 255 characters)
    max_length = 80
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def process_and_align(download_directory):
    try:
        # Clean the transcripts
        clean_directory(download_directory)   
        convert_mp3_to_wav(download_directory)

        acoustic_model = 'english_us_arpa'
        dictionary = 'english_us_arpa'
        beam = 10
        retry_beam = 40
        num_jobs = 14

        run_mfa_align(download_directory, acoustic_model, dictionary, output_path, beam, retry_beam, num_jobs)
        # don't convert for now
        # convert_textgrids_to_csv(download_directory) 

    except Exception as e:
        print(f"process_and_align failed to run: {e}")

                               
def get_json_objects_batch(start_index, batch_size=5):
    unique_id = os.getpid()  # Unique process ID
    download_directory = f"{download_path}_{start_index}_{unique_id}"
    os.makedirs(download_directory, exist_ok=True)
    downloaded_mp3 = False
    
    try:
        # Open the file for appending URLs
        with open("/shared/3/projects/bangzhao/prosodic_embeddings/mfa/enclosures.txt", "a", encoding="utf-8") as url_file:
            with open(metadata_path, 'r', encoding='utf-8') as file:
                current_line_num = 0

                # Skip lines until the start_index
                while current_line_num < start_index:
                    line = file.readline()
                    if not line:
                        break
                    current_line_num += 1

                # Process the batch
                for line_num in range(batch_size):
                    line = file.readline()
                    current_line_num += 1

                    if not line:
                        break  # End of file reached

                    try:
                        json_object = json.loads(line)
                        # filename = 'test'
                        # filename = os.path.splitext(os.path.basename(json_object['potentialOutPath']))[0]
                        filename = sanitize_filename(json_object['enclosure'])
                        filename = f"{current_line_num}-{filename}"

                        # Store the URL
                        url_file.write(f"{json_object['enclosure']}\n")

                        success = download_mp3(json_object, filename, download_directory)
                        if success:
                            downloaded_mp3 = True
                        download_transcript(json_object, filename, download_directory)

                    except Exception as e:
                        print(f"An error occurred at line {current_line_num}: {e}")
        
        # Run MFA only when MP3 files have been downloaded
        if downloaded_mp3:
            process_and_align(download_directory)
            print(f"Batch starting at {start_index} processed.")
        else:
            print(f"No audio file in cache_{start_index}")

        time.sleep(5)
    
    finally:
        # Clean up the download directory
        shutil.rmtree(download_directory, ignore_errors=True)
        print(f"Successfully removed directory: {download_directory}")


    
def parallel_process(total_rows, start_line=0, num_processes=12, batch_size=5):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for start_index in range(start_line, total_rows, batch_size):
            futures.append(executor.submit(get_json_objects_batch, start_index, batch_size))

        # Optional: Monitor the progress
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get the result or raise an exception if occurred
            except Exception as e:
                print(f"Error processing batch: {e}")

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON objects in parallel.")
    
    parser.add_argument('--start_line', type=int, default=0, help="The line to start processing from.")
    parser.add_argument('--num_processes', type=int, default=12, help="The number of processes to run in parallel.")
    parser.add_argument('--batch_size', type=int, default=5, help="The size of each batch of tasks.")

    args = parser.parse_args()

    total_rows = 1124058  # Set your total rows here or calculate it dynamically
    parallel_process(total_rows, start_line=args.start_line, num_processes=args.num_processes, batch_size=args.batch_size)
    
    
    
    
# def parse_textgrid(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     tier_name = None
#     tier_type = None
#     start_time = None
#     end_time = None
#     label = None
#     in_intervals = False

#     for line in lines:
#         line = line.strip()
        
#         if line.startswith('name = '):
#             tier_name = line.split('=')[1].strip().strip('"')
#         elif line.startswith('class = "IntervalTier"'):
#             tier_type = 'IntervalTier'
#         elif line.startswith('class = "TextTier"'):
#             tier_type = 'TextTier'
#         elif line.startswith('intervals:'):
#             in_intervals = True
#         elif line.startswith('points:'):
#             in_intervals = False
#         elif in_intervals and line.startswith('xmin ='):
#             start_time = float(line.split('=')[1].strip())
#         elif in_intervals and line.startswith('xmax ='):
#             end_time = float(line.split('=')[1].strip())
#         elif in_intervals and line.startswith('text ='):
#             label = line.split('=')[1].strip().strip('"')
#             data.append([tier_name, start_time, end_time, label])
#         elif tier_type == 'TextTier' and line.startswith('number ='):
#             time = float(line.split('=')[1].strip())
#             label = 'point'  # Placeholder, replace with actual point data if needed
#             data.append([tier_name, time, time, label])

#     return data


# def convert_textgrids_to_csv(directory):
#     # Iterate over all files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith(".TextGrid"):
#             textgrid_path = os.path.join(directory, filename)
#             csv_path = os.path.splitext(textgrid_path)[0] + '.csv'

#             # Parse the TextGrid file
#             data = parse_textgrid(textgrid_path)

#             # Create a DataFrame
#             df = pd.DataFrame(data, columns=['Tier', 'Start Time', 'End Time', 'Label'])

#             # Save the DataFrame to a CSV file
#             df.to_csv(csv_path, index=False)
#             print(f"Converted {textgrid_path} to {csv_path}")

#             # Remove the original TextGrid file
#             os.remove(textgrid_path)
#             print(f"Removed original TextGrid file: {textgrid_path}")


# def count_rows(file_path):
#     """Count the total number of rows in a JSONL file."""
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return sum(1 for _ in file)