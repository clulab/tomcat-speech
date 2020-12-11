
// Create WebSocket connection
const socket = new WebSocket('ws://localhost:9000')
socket.addEventListener('open',
    function(event) { socket.send("Message from browser"); });

// Listen for messages
socket.addEventListener(
    "message",
    function(event) { console.log("message from server ", event.data); });
// navigator.mediaDevices is a singleton object of type MediaDevices.
if (navigator.mediaDevices) {
    navigator.mediaDevices
        .getUserMedia({video : true, audio : false})
        .then(function onSuccess(stream) {
            // stream is an object of type MediaStream
            const video = document.getElementById('webcam');
            video.autoplay = true;
            video.srcObject = stream;

            // The MediaRecorder interface takes the data from a MediaStream
            // delivers it to you for processing.
            // https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API
            const recorder = new MediaRecorder(stream);
            const data = [];

            // Register a function to handle data.
            recorder.ondataavailable =
                function(event) {
                    // e.data is a Blob object that contains the media data.
                    console.log("data is available!")
                    socket.send(event.data)
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
