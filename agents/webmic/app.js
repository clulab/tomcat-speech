// Code adapted from
// https://github.com/vin-ni/Google-Cloud-Speech-Node-Socket-Playground

'use strict'

// Edit the 'config' object below if you want to set a different destination
// host/port.
var config = {
    "destination_host": "localhost",
    "destination_port": 8888
}

// Get parameters from URL query string
const params = new URLSearchParams(window.location.search);
const participantId = params.get("id");

function makeSocket(destination) {
    var ws = new WebSocket(destination); 

    // Listen for messages
    ws.onopen = function(event) {
        document.getElementById("connectedIndicator").innerHTML="Connected";

        var connectButton = document.getElementById("connectButton");
        connectButton.disabled = true;
        console.log("Websocket connected!")
    }

    ws.onmessage = function(event) {
        document.getElementById("participantId").innerHTML="Participant ID: " + event.data;
    };

    ws.onerror = (error) => {
        console.error("Error in creating websocket. Trying again in 5 seconds.");
        ws.close();
    };

    ws.onclose = function(event) {
        document.getElementById("connectedIndicator").innerHTML="Not connected";
        setTimeout(function() {
            ws = makeSocket(destination);
        }, 5000);
    };

    return ws;
}

class PersistentSocket {
    constructor(destination){
        this.destination = destination
        this.ws = makeSocket(this.destination);
    }

    send(data) {
        if (this.ws.readyState == 1) {
            this.ws.send(data);
        }
    }
}

let socket,
    sampleRate;

document.getElementById("connectButton").onclick = function() {
    var context = getAudioContext();
    var destination = "ws://"+config["destination_host"]+":"+config["destination_port"].toString()
                        + "?id="+participantId + "&sampleRate=" + context.sampleRate;
    socket=new PersistentSocket(destination);
    initRecording(context);
};

//================= CONFIG =================
// Stream Audio
let AudioContext,
	context,
	processor,
	input,
	globalStream;

//vars
let audioElement = document.querySelector('audio'),
	finalWord = false,
	resultText = document.getElementById('ResultText'),
	removeLastSentence = true,
	streamStreaming = false;


//audioStream constraints
const constraints = {
	audio: true,
	video: false
};

//================= RECORDING =================



function getAudioContext() {
	AudioContext = window.AudioContext || window.webkitAudioContext;
	context = new AudioContext({
        // if Non-interactive, use 'playback' or 'balanced'
        // https://developer.mozilla.org/en-US/docs/Web/API/AudioContextLatencyCategory
		latencyHint: 'interactive',
	});
    return context;
}

function initRecording(context) {
    streamStreaming = true;
	processor = context.createScriptProcessor(0, 1, 1);
	processor.connect(context.destination);
	context.resume();

	var handleSuccess = function (stream) {
		globalStream = stream;
		input = context.createMediaStreamSource(stream);
		input.connect(processor);

		processor.onaudioprocess = function (e) {
			microphoneProcess(e);
		};
	};

	navigator.mediaDevices.getUserMedia(constraints)
		.then(handleSuccess);

}

function microphoneProcess(e) {
	var channelData = e.inputBuffer.getChannelData(0);
    socket.send(channelData);
}
