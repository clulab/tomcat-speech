// Code adapted from
// https://github.com/vin-ni/Google-Cloud-Speech-Node-Socket-Playground

'use strict'

function makeSocket(destination, subprotocol) {
    var ws = new WebSocket(destination, subprotocol); 

    // Listen for messages
    ws.onopen = function(event) {
        document.getElementById("connectedIndicator").innerHTML="Connected";

        var connectButton = document.getElementById("connectButton");
        connectButton.disabled = true;
        console.log("Websocket connected!")
    }

    ws.onmessage = function(event) {
        console.log("Message received from server", event.data);
    };

    ws.onerror = (error) => {
        console.error("Error in creating websocket. Trying again in 5 seconds.");
        ws.close();
    };

    ws.onclose = function(event) {
        document.getElementById("connectedIndicator").innerHTML="Not connected";
        setTimeout(function() {
            ws = makeSocket(destination, subprotocol);
        }, 5000);
    };

    return ws;
}

class PersistentSocket {
    constructor(destination, subprotocol){
        this.destination = destination
        this.subprotocol = subprotocol
        this.ws = makeSocket(this.destination, this.subprotocol);
    }

    send(data) {
        if (this.ws.readyState == 1) {
            this.ws.send(data);
        }
    }
}

let socket;

document.querySelector("form").addEventListener("submit", (e) => {
    const formData = new FormData(e.target);
    var subprotocol=formData.get("participant_id");
    var destination = "ws://localhost:8000";
    socket=new PersistentSocket(destination, subprotocol);
    e.preventDefault();
});


//================= CONFIG =================
// Stream Audio
let bufferSize = 2048,
	AudioContext,
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



function initRecording() {
	streamStreaming = true;
	AudioContext = window.AudioContext || window.webkitAudioContext;
	context = new AudioContext({
        // if Non-interactive, use 'playback' or 'balanced'
        // https://developer.mozilla.org/en-US/docs/Web/API/AudioContextLatencyCategory
		latencyHint: 'interactive',
	});
	processor = context.createScriptProcessor(bufferSize, 1, 1);
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


//================= INTERFACE =================
var startButton = document.getElementById("startRecButton");
startButton.addEventListener("click", startRecording);

var endButton = document.getElementById("stopRecButton");
endButton.addEventListener("click", stopRecording);
endButton.disabled = true;

function startRecording() {
	startButton.disabled = true;
	endButton.disabled = false;
	initRecording();
}

function stopRecording() {
	// waited for FinalWord
	startButton.disabled = false;
	endButton.disabled = true;
	streamStreaming = false;


	let track = globalStream.getTracks()[0];
	track.stop();

	input.disconnect(processor);
	processor.disconnect(context.destination);
	context.close().then(function () {
		input = null;
		processor = null;
		context = null;
		AudioContext = null;
		startButton.disabled = false;
	});
}
