import boto3
import os
import sys
from time import sleep
import requests


"""
To run this script you need a config file at:
~/.aws/config
The file should look like:

[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
"""

S3_BUCKET = "az-tomcat-project"
S3_PATH = "hackathon"


def transcribe_file(file_path, output_dir_path="."):
    transcribe_client = boto3.client("transcribe")
    cur_dir = os.path.abspath(file_path)
    files_to_upload = []
    if os.path.isdir(cur_dir):
        for cur in os.listdir(cur_dir):
            cur_name = os.fsdecode(cur)
            if cur_name[-4:] != ".wav":
                continue
            files_to_upload.append(os.path.join(file_path, cur))
    else:
        if file_path[-4:] == ".wav":
            files_to_upload.append(file_path)
        else:
            print("First arg must be directory or .wav file")
            return

    if len(files_to_upload) < 1:
        print("No .wav files found, exitting...")
        return

    for transcribe_file in files_to_upload:
        file_name = str(transcribe_file).split("/")[-1]
        job_name = "transcription_" + file_name
        print(f"Starting transcription of: {S3_PATH}/{file_name}")
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": f"s3://{S3_BUCKET}/{S3_PATH}/{file_name}"},
            MediaFormat="wav",
            LanguageCode="en-US",
        )

        transcription_filepath = f"{output_dir_path}/{job_name[:-4]}_transcript"

        while True:
            status = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
            if job_status in ["COMPLETED", "FAILED"]:
                break
            sleep(1)
        result_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

        req = requests.get(url=result_url)
        result_json = req.json()
        with open(transcription_filepath + "_full.txt", "w") as out_file:
            out_file.write(str(result_json))
        with open(transcription_filepath + ".txt", "w") as out_file:
            out_file.write(str(result_json["results"]["transcripts"][0]["transcript"]))
        print(f"Collected transcript for file {job_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Insufficient arguments, usage: python3 aws_script.py input_dir [output_dir]"
        )
    elif len(sys.argv) > 2:
        transcribe_file(sys.argv[1], sys.argv[2])
    else:
        transcribe_file(sys.argv[1])
