// document.addEventListener("DOMContentLoaded", function () {
//     let recordBtn = document.getElementById("record-btn");
//     let status = document.getElementById("status");
//     let audioPlayer = document.getElementById("audio-player");
//     let uploadForm = document.getElementById("upload-form");
//     let fileInput = document.getElementById("audio-file");
//     let resultDisplay = document.getElementById("result");

//     let csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute("content");

//     let mediaRecorder;
//     let audioChunks = [];
//     let isRecording = false;

//     recordBtn.addEventListener("click", async () => {
//         let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//         mediaRecorder = new MediaRecorder(stream);

//         mediaRecorder.ondataavailable = (event) => {
//             audioChunks.push(event.data);
//         };

//         mediaRecorder.onstop = () => {
//             let audioBlob = new Blob(audioChunks, { type: "audio/wav" });
//             let audioURL = URL.createObjectURL(audioBlob);
//             audioPlayer.src = audioURL;
//             uploadForm.style.display = "block";

//             let file = new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" });
//             let dataTransfer = new DataTransfer();
//             dataTransfer.items.add(file);
//             fileInput.files = dataTransfer.files;
//         };

//         audioChunks = [];
//         mediaRecorder.start();
//         status.innerText = "Recording... Speak now!";
//         setTimeout(() => {
//             mediaRecorder.stop();
//             status.innerText = "Recording stopped. Play and analyze!";
//         }, 5000);
//     });

//     uploadForm.addEventListener("submit", async (event) => {
//         event.preventDefault();
//         let formData = new FormData();
//         formData.append("audio", fileInput.files[0]);

//         let response = await fetch("/analyze_voice/", {
//             method: "POST",
//             headers: {
//                 "X-CSRFToken": csrfToken,  // Add CSRF token in the headers
//             },
//             body: formData,
//         });

//         let data = await response.json();
//         resultDisplay.innerText = data.emotion;
//     });
// });


document.addEventListener("DOMContentLoaded", function () {
    let recordBtn = document.getElementById("record-btn");
    let status = document.getElementById("status");
    let audioPlayer = document.getElementById("audio-player");
    let uploadForm = document.getElementById("upload-form");
    let fileInput = document.getElementById("audio-file");
    let resultDisplay = document.getElementById("result");
    let csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute("content");

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false; // Add a flag to track recording state

    recordBtn.addEventListener("click", async () => {
        if (isRecording) {
            // If already recording, stop the recording
            mediaRecorder.stop();
            isRecording = false; // Update recording state
            recordBtn.textContent = "🎙️ Record (5s)"; // Change button text back
            status.innerText = "Recording stopped. Play and analyze!";
            return;
        }

        try {
            let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            // mediaRecorder = new MediaRecorder(stream,{ mimeType: 'audio/wav' });

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                let audioBlob = new Blob(audioChunks, { type: "wav" });
                let audioURL = URL.createObjectURL(audioBlob);
                audioPlayer.src = audioURL;
                uploadForm.style.display = "block";

                let file = new File([audioBlob], "recorded_audio.wav", { type: "wav" });
                let dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                isRecording = false;
            };

            mediaRecorder.start();
            isRecording = true; // Update recording state
            recordBtn.textContent = "Stop Recording";
            status.innerText = "Recording... Speak now!";
            audioChunks = [];
            
            setTimeout(() => {
                if (isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    recordBtn.textContent = "🎙️ Record (5s)";
                    status.innerText = "Recording stopped. Play and analyze!";
                }
            }, 2000);
        } catch (error) {
            console.error("Error accessing microphone:", error);
            status.innerText = "Error accessing microphone.";
            isRecording = false; // Ensure recording state is reset
        }
    });

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        let formData = new FormData();
        formData.append("audio", fileInput.files[0]);

        let response = await fetch("/analyze_voice/", {
            method: "POST",
            headers: {
                "X-CSRFToken": csrfToken,
            },
            body: formData,
        });

        if (response.ok) {
            let data = await response.json();
            resultDisplay.innerText = data.emotion;
        } else {
            resultDisplay.innerText = "Error analyzing audio";
        }
    });
});




// Ensure Recorder.js is loaded before this script
// document.addEventListener("DOMContentLoaded", function () {
//     let recordBtn = document.getElementById("record-btn");
//     let status = document.getElementById("status");
//     let audioPlayer = document.getElementById("audio-player");
//     let uploadForm = document.getElementById("upload-form");
//     let fileInput = document.getElementById("audio-file");
//     let resultDisplay = document.getElementById("result");
//     let csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute("content");

//     let recorder; 
//     let audioContext;
//     let gumStream;

//     recordBtn.addEventListener("click", async () => {
//       // Request access to the microphone
//     try {
//         gumStream = await navigator.mediaDevices.getUserMedia({ audio: true });
//         audioContext = new AudioContext();
//         let input = audioContext.createMediaStreamSource(gumStream);
//         // Initialize Recorder.js with one channel (mono)
//         recorder = new Recorder(input, { numChannels: 1 });
//         recorder.record();
//         status.innerText = "Recording... Speak now!";
//         // Stop recording after 5 seconds
//         setTimeout(() => {
//             recorder.stop();
//           // Stop all audio tracks to release the microphone
//             gumStream.getAudioTracks().forEach(track => track.stop());
//             status.innerText = "Recording stopped. Processing...";
//           // Export the recording as a WAV blob
//             recorder.exportWAV(function (blob) {
//             let audioURL = URL.createObjectURL(blob);
//             audioPlayer.src = audioURL;
//             uploadForm.style.display = "block";
//             status.innerText = "Ready to analyze!";
            
//             // Convert Blob to File
//             let file = new File([blob], "recorded_audio.wav", { type: "audio/wav" });
//             let dataTransfer = new DataTransfer();
//             dataTransfer.items.add(file);
//             fileInput.files = dataTransfer.files;
            
//             // Clear recorder for future recordings
//             recorder.clear();
//         });
//         }, 5000);
//     } catch (err) {
//         console.error("Error accessing microphone:", err);
//         status.innerText = "Error accessing microphone.";
//     }
//     });
//     uploadForm.addEventListener("submit", async (event) => {
//         event.preventDefault();
//         let formData = new FormData();
//         formData.append("audio", fileInput.files[0]);
//         let response = await fetch("/analyze_voice/", {
//             method: "POST",
//             headers: {
//                 "X-CSRFToken": csrfToken,
//             },
//             body: formData,
//         });
//         let data = await response.json();
//         resultDisplay.innerText = data.emotion;
//     });
// });