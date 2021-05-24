import subprocess
import os
import sys
import boto3
from botocore.exceptions import ClientError

"""
To run this script you need a config file at:
~/.aws/config
The file should look like:

[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
"""

S3_BUCKET = "az-tomcat-project"  # CHANGE THIS
S3_PATH = "hackathon"  # CHANGE THIS


def upload_file(upload_file_path):
    s3 = boto3.client("s3")
    files_to_upload = []
    if os.path.isdir(upload_file_path):
        for cur in os.listdir(upload_file_path):
            cur_name = os.fsdecode(cur)
            if cur_name[-4:] != ".mp4":
                continue
            files_to_upload.append(os.path.join(upload_file_path, cur))
    else:
        if upload_file_path[-4:] != ".mp4":
            print("File is not an mp4 video file")
            return
        else:
            files_to_upload.append(upload_file_path)

    if len(files_to_upload) < 1:
        print("No video files found for upload")
        return

    for file_to_upload in files_to_upload:
        file_name = str(file_to_upload).split("/")[-1]
        print("Working on " + str(file_to_upload))

        # Convert video to wav
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                f"{file_to_upload}",
                "-vn",
                f"{file_to_upload[:-4]}.wav",
            ]
        )

        # Upload file to AWS
        if not os.path.exists(f"{file_to_upload[:-4]}.wav"):
            raise FileNotFoundError(file_to_upload)
        print(f" Uploading file: {file_to_upload[:-4]}.wav ...")
        save_name = f"{file_name[:-4]}.wav"
        try:
            response = s3.upload_file(
                file_to_upload[:-4] + ".wav",
                S3_BUCKET,
                S3_PATH + "/" + save_name,
            )
        except ClientError as e:
            print(e)
            raise
        print("    Done")


if __name__ == "__main__":
    assert len(sys.argv) > 1
    upload_file(sys.argv[1])
