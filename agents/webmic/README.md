webmic
======

A webapp to capture microphone audio and stream it to a server.

Quickstart
----------

To run the app with Python 3's built-in HTTP server, run the following command
from within this directory:

    python -m http.server

Then navigate to http://localhost:8000 in your browser.

Docker instructions
-------------------

We also provide a Dockerfile for easy deployment.

To build the container, do:

    docker build -t webmic .
    
To run the container, do:

    docker run -p 8000:8000 webmic
