from python:3.7
copy requirements.txt .
run pip install -r requirements.txt
copy asrserver .
expose 8000
env HOST=0.0.0.0
entrypoint ["./asrserver"]
