// recorder.js
(function(window){

    // Path to the worker file – adjust the path as needed (e.g., in your static folder)
    var WORKER_PATH = 'recorder_worker.js';
    var Recorder = function(source, cfg){
    var config = cfg || {};
    var bufferLen = config.bufferLen || 4096;
    this.context = source.context;
    this.node = (this.context.createScriptProcessor ||
                    this.context.createJavaScriptNode).call(this.context,bufferLen, config.numChannels || 2, config.numChannels || 2);
    
      // Initialize the worker
    this.worker = new Worker(config.workerPath || WORKER_PATH);
    this.worker.postMessage({
        command: 'init',
        config: {
            sampleRate: this.context.sampleRate,
            numChannels: config.numChannels || 2
        }
    });
    var recording = false,
    currCallback;
    this.node.onaudioprocess = function(e){
        if (!recording) return;
            var buffer = [];
            for (var channel = 0; channel < (config.numChannels || 2); channel++){
            buffer.push(e.inputBuffer.getChannelData(channel));
        }
        // Send audio data to the worker
        _this.worker.postMessage({
        command: 'record',
        buffer: buffer
    });
    };
      // To preserve "this" inside the onaudioprocess function
    var _this = this;

      // Start recording
    this.record = function(){
        recording = true;
    };
      // Stop recording
    this.stop = function(){
        recording = false;
    };
      // Clear all recorded data
    this.clear = function(){
        this.worker.postMessage({ command: 'clear' });
    };
      // Export the recorded audio as a WAV blob
    this.exportWAV = function(cb, type){
        currCallback = cb || config.callback;
        type = type || 'audio/wav';
        this.worker.postMessage({
            command: 'exportWAV',
            type: type
        });
        this.worker.onmessage = function(e){
            var blob = e.data;
            currCallback(blob);
        };
    };
      // Connect the node to the audio graph so that it starts processing.
    source.connect(this.node);
      // Optionally, if you want playback during recording (remove if not desired)
    this.node.connect(this.context.destination);
    };

    // Expose Recorder to the global object
    window.Recorder = Recorder;
})(window);
