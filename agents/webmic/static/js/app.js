// Code adapted from
// https://github.com/vin-ni/Google-Cloud-Speech-Node-Socket-Playground

'use strict'

class PersistentSocket {
    constructor(destination){
        this.ws = new WebSocket(destination); 
        // Listen for messages
        this.ws.onmessage = function(event) {
            console.log("Message received from server", event.data);
        };

        this.ws.onclose = function(event) {
            console.log("Socket closed, trying again in 1 second.", event.data);
            setTimeout(function(){
                this.ws = new WebSocket(destination);
            }, 1000)
        };
    }

    send(data) {
        this.ws.send(data);
    }
}

let socket;

document.querySelector("form").addEventListener("submit", (e) => {
    const formData = new FormData(e.target);
    var host=formData.get("host");
    var port=formData.get("port");
    socket=new PersistentSocket("ws://"+host+":"+port);
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

var recordingStatus = document.getElementById("recordingStatus");

function startRecording() {
	startButton.disabled = true;
	endButton.disabled = false;
	//recordingStatus.style.visibility = "visible";
	initRecording();
}

function stopRecording() {
	// waited for FinalWord
	startButton.disabled = false;
	endButton.disabled = true;
	recordingStatus.style.visibility = "hidden";
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
