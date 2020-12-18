from python:3.7-alpine
run pip install flask
env FLASK_APP=webmic.py
env FLASK_RUN_HOST=0.0.0.0
expose 5000
copy . .
entrypoint ["flask", "run"]
