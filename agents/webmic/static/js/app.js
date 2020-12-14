// Create WebSocket connection
const socket = new WebSocket('ws://localhost:9000')
socket.onopen= function(event) { console.log("Socket opened."); };

// Listen for messages
socket.onmessage = function(event) { console.log("message from server ", event.data); };
socket.onclose = function(event) { console.log("message from server ", event.data); };

// navigator.mediaDevices is a singleton object of type MediaDevices.
if (navigator.mediaDevices) {
    navigator.mediaDevices
        .getUserMedia({video : false, audio : true})
        .then(function onSuccess(stream) {
            // stream is an object of type MediaStream

            // The MediaRecorder interface takes the data from a MediaStream
            // delivers it to you for processing.
            // https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API
            const recorder = new MediaRecorder(stream);

            // Register a function to handle data.
            recorder.ondataavailable =
                function(event) {
                    // event.data is a Blob object that contains the media data.
                    socket.send(event.data);
                }

            // Register a function to handle errors.
            recorder.onerror =
                function(event) {
                    // e.name is FF non-spec
                    throw event.error || new Error(event.name); 
                }

            // Specify 100ms time slices
            recorder.start(timeslice=100);
        })
        .catch(function onError() {
            alert(
                'There has been a problem retreiving the streams - are you running on file:/// or did you disallow access?');
        });
}
else {
    alert('getUserMedia is not supported in this browser.');
}
