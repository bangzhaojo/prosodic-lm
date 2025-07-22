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
                

def run_mfa_align(input_path, acoustic_model, dictionary, output_path, beam, retry_beam, num_jobs, timeout_duration="80m", max_retries=5):
    # Generate a unique command history file path based on the process ID
    
    cache_file = os.path.basename(input_path)
    temp_directory = f'/shared/3/projects/bangzhao/prosodic_embeddings/mfa/temp/{cache_file}'
    os.makedirs(temp_directory, exist_ok=True)
    
    unique_id = os.getpid()
    
    # Construct the MFA align command as a single string
    command = (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate mfa && "
        f"mfa align --clean --output_format csv --final_clean "
        f"--temporary_directory {temp_directory} {input_path} {acoustic_model} {dictionary} "
        f"{output_path} --beam {beam} --retry_beam {retry_beam} --num_jobs {num_jobs} --single_speaker"
    )

    # Include the timeout command
    full_command = f"timeout {timeout_duration} bash -c '{command}'"

    # Print the command to verify
    print("Running command:", full_command)

    # Attempt to run the command with retries for non-timeout errors
    attempt = 0
    while attempt < max_retries:
        try:
            # Run the command using the shell
            subprocess.run(full_command, shell=True, check=True, executable='/bin/bash')
            print("Command executed successfully.")
            break  # If successful, exit the loop
        except subprocess.CalledProcessError as e:
            if e.returncode == 124:
                print("Command timed out.")
                break  # Do not retry if the command timed out
            else:
                attempt += 1
                print(f"Failed to execute command on attempt {attempt}: {e}")
                
                if attempt < max_retries:
                    print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                else:
                    print("Max retries reached. Command failed.")

    # Clean up the command history file after the command execution
    if os.path.exists(command_history_file):
        os.remove(command_history_file)
        print(f"Removed command history file: {command_history_file}")

    # Remove files or directories under the temp directory
    if os.path.exists(temp_directory):
        shutil.rmtree(temp_directory)
        print(f"Removed temporary directory: {temp_directory}")

    # Notify the completion of the process
    print(f"MFA alignment completed for {input_path}, output available at {output_path}")
        

def sanitize_filename(url):
    # Decode the URL-encoded characters to handle %20, etc.
    url = urllib.parse.unquote(url)
    
    # Replace illegal characters with underscores or remove them
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '', url)
    
    # Remove any trailing periods or spaces (illegal in Windows filenames)
    sanitized = sanitized.strip().rstrip('.')

    # Optionally, truncate the filename if it's too long (common limit is 255 characters)
    max_length = 128
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def download_and_convert(json_object, line_number, download_directory):
    """Download MP3, save transcript, and convert MP3 to WAV."""
    mp3_url = json_object['enclosure']
    transcript = json_object["transcript"]
    sanitized_filename = sanitize_filename(mp3_url)

    # Construct the base filename
    base_filename = f"{line_number}-{sanitized_filename}"
    
    try:
        # Download MP3
        mp3_filename = os.path.join(download_directory, f"{base_filename}.mp3")
        response = requests.get(mp3_url, stream=True)
        if response.status_code == 200:
            with open(mp3_filename, 'wb') as mp3_file:
                for chunk in response.iter_content(chunk_size=8192):
                    mp3_file.write(chunk)
            # print(f"Downloaded MP3: {mp3_filename}")
        else:
            # print(f"Failed to download MP3 {mp3_url}, status code: {response.status_code}")
            return False

        # Save Transcript
        transcript_filename = os.path.join(download_directory, f"{base_filename}.txt")
        with open(transcript_filename, 'w', encoding='utf-8') as transcript_file:
            transcript_file.write(transcript)
        # print(f"Transcript saved as {transcript_filename}")

        # Convert MP3 to WAV
        wav_filename = os.path.splitext(mp3_filename)[0] + '.wav'
        command = ['ffmpeg', '-y', '-i', mp3_filename, wav_filename]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(mp3_filename)
        # print(f"Converted MP3 to WAV and removed original MP3: {mp3_filename}")

        return True

    except Exception as e:
        # print(f"Failed to process {line_number}: {e}")
        return False

    
def process_batch_files(json_objects, start_index, download_directory):
    # Use multiprocessing Pool to download MP3, save transcripts, and convert to WAV in parallel
    with Pool(processes=8) as pool:
        results = pool.starmap(download_and_convert, [(json_object, start_index + i, download_directory) for i, json_object in enumerate(json_objects)])

        
def process_and_align(download_directory):
    try:
        # Assuming clean_directory is defined elsewhere
        clean_directory(download_directory)   

        acoustic_model = 'english_us_arpa'
        dictionary = 'english_us_arpa'
        beam = 10
        retry_beam = 40
        num_jobs = 8

        run_mfa_align(download_directory, acoustic_model, dictionary, output_path, beam, retry_beam, num_jobs)
        print(f"Alignment process completed for {download_directory}")
        
    except Exception as e:
        print(f"process_and_align failed to run: {e}")

                               
def get_json_objects_batch(start_index, batch_size=5):
    unique_id = os.getpid()  # Unique process ID
    download_directory = f"{download_path}_{start_index}_{unique_id}"
    os.makedirs(download_directory, exist_ok=True)
    
    try:
        json_objects = []
        with open(metadata_path, 'r', encoding='utf-8') as file, \
             open("/shared/3/projects/bangzhao/prosodic_embeddings/mfa/enclosures.txt", "a", encoding="utf-8") as url_file:
            current_line_num = 0

            # Skip lines until the start_index
            while current_line_num < start_index:
                line = file.readline()
                if not line:
                    break
                current_line_num += 1

            # Collect batch of JSON objects
            for line_num in range(batch_size):
                line = file.readline()
                current_line_num += 1

                if not line:
                    break  # End of file reached

                try:
                    json_object = json.loads(line)
                    json_objects.append(json_object)

                    # Record the URL in enclosures.txt
                    url_file.write(f"{json_object['enclosure']}\n")

                except Exception as e:
                    print(f"An error occurred at line {current_line_num}: {e}")
        
        # Download MP3, save transcripts, and convert to WAV in parallel
        process_batch_files(json_objects, start_index, download_directory)
        
        process_and_align(download_directory)
        print(f"Batch starting at {start_index} processed.")
        
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