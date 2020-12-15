"use strict"
    // Code adapted from
    // https://github.com/vin-ni/Google-Cloud-Speech-Node-Socket-Playground

class AudioStreamer {
    constructor() {
        this.bufferSize = 2048;
        this.constraints = {audio : true, video : false};
        // We may want to add || window.webkitAudioContext to support older
        // browsers?
        this.context = new window.AudioContext({latencyHint : "interactive"});
        this.processor =
            this.context.createScriptProcessor(this.bufferSize, 1, 1);
        this.processor.connect(this.context.destination);
        this.context.resume();
        this.makeSocket();

        navigator.mediaDevices.getUserMedia(this.constraints)
            .then(function(stream) {
                this.input = this.context.createMediaStreamSource(stream);
                this.input.connect(this.processor);
                this.processor.onaudioprocess = function(e) {
                    this.microphoneProcess(e);
                };
            });
    }

    // Create WebSocket connection
    makeSocket() {
        this.socket = new WebSocket("ws://localhost:9000");
        this.socket.onopen = function(event) { console.log("Socket opened."); };
        // Listen for messages
        this.socket.onmessage = function(event) {
            console.log("Message received from server", event.data);
        };
        this.socket.onclose = function(event) {
            console.log("Socket closed", event.data);
        };
    }

    microphoneProcess(e) {
        var left = e.inputBuffer.getChannelData(0);
        this.socket.send(e.data);
    }
}

var startButton = document.getElementById("startRecButton");
startButton.addEventListener(
    "click", function() { const streamer = new AudioStreamer(); });
