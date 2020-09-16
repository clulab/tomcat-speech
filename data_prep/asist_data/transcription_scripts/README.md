# Transcription Script Overview

## AWS
There are two AWS scripts that need to be used:
1. aws_upload.py
2. aws_script.py

The first uploads files to a S3 bucket and the second attempts to transcribe
them with AWS. There are two scripts so that if something breaks during one
step, the process does not need to be entirely restarted. Both use a hardcoded
S3 bucket, whose value must be changed for reuse.

### Input / Output
The scripts both take filepaths as arguments. The first requires that you pass
a file path or a directory path where the video files are stored. The second
requires that you pass in the same first argument but has an optional second
argument for a location to output transcriptions to.

The second script will look at all the .wav files in the given filepath and try
to transcribe files that look like them from the S3 bucket. If the files are
transcribed successfully, two output files will be saved. One with the entire
JSON returned from AWS and the other with just the transcription text. The file
with the entire JSON looks the same as the transcript but adds an additional
`_full` to the end of the file name.

### Running the scripts
To run the scripts you need to configure your AWS credentials in your home
directory. You will need to create a file in `~/.aws/config` that looks like:

    [default]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY

For the AWS upload to work, the key you use needs read/write permission to the
bucket hardcoded in the two scripts. You should change this bucket to fit your
needs. If that's properly configured, you'll just need to run each script in
succession. Here are some example invocations of the scripts:

    python3 aws_upload.py saved_data_dir/

Followed by:

    python3 aws_script.py saved_data_dir/ transcription_output/
