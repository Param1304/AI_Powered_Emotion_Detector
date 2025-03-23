document.addEventListener("DOMContentLoaded", function () {
    const recordBtn = document.getElementById("record-btn");
    const status = document.getElementById("status");
    const audioPlayer = document.getElementById("audio-player");
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("audio-file");
    const resultDisplay = document.getElementById("result");

    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute("content");

    let mediaRecorder;
    let audioChunks = [];

    recordBtn.addEventListener("click", async () => {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioURL = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioURL;
            uploadForm.style.display = "block";
            const file = new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
        };

        // Clear previous data
        audioChunks = [];
        mediaRecorder.start();
        status.innerText = "Recording... Speak now!";
        
        // Set timeout for 5 seconds (5000 milliseconds)
        setTimeout(() => {
            mediaRecorder.stop();
            status.innerText = "Recording stopped. Play and analyze!";
        }, 5000); // 5000 ms = 5 seconds
    });
    
    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append("audio", fileInput.files[0]);
        const response = await fetch("/analyze_voice/", {
            method: "POST",
            headers: {
                "X-CSRFToken": csrfToken,
            },
            body: formData,
        });

        const data = await response.json();
        resultDisplay.innerText = data.emotion;
    });
});





