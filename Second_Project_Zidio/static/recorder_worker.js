// recorderWorker.js
var recLength = 0,
    recBuffers = [],
    sampleRate,
    numChannels;

self.onmessage = function(e){
/* This code snippet is a switch statement that is handling different commands received via the
`onmessage` event in a web worker. When a message is received, the switch statement checks the value
of `e.data.command` and executes the corresponding function based on the command received. Here's
what each case does: */
    switch(e.data.command){
        case 'init':
        init(e.data.config);
        break;
        case 'record':
        record(e.data.buffer);
        break;
        case 'exportWAV':
        exportWAV(e.data.type);
        break;
        case 'clear':
        clear();
        break;
    }
    };

function init(config){
    sampleRate = config.sampleRate;
    numChannels = config.numChannels;
    initBuffers();
}

function record(inputBuffer){
for (var channel = 0; channel < numChannels; channel++){
    recBuffers[channel].push(inputBuffer[channel]);
}
    recLength += inputBuffer[0].length;
}

function exportWAV(type){
    var buffers = [];
    for (var channel = 0; channel < numChannels; channel++){
        buffers.push(mergeBuffers(recBuffers[channel], recLength));
    }
    var interleaved;
    if (numChannels === 2){
        interleaved = interleave(buffers[0], buffers[1]);
    } else {
        interleaved = buffers[0];
    }
    var dataview = encodeWAV(interleaved);
    var audioBlob = new Blob([dataview], { type: type });
    self.postMessage(audioBlob);
}

function initBuffers(){
    recBuffers = [];
    for (var channel = 0; channel < numChannels; channel++){
        recBuffers[channel] = [];
    }
    recLength = 0;
}

function mergeBuffers(recBuffers, recLength){
    var result = new Float32Array(recLength);
    var offset = 0;
    for (var i = 0; i < recBuffers.length; i++){
        result.set(recBuffers[i], offset);
        offset += recBuffers[i].length;
    }
    return result;
}

function interleave(inputL, inputR){
    var length = inputL.length + inputR.length;
    var result = new Float32Array(length);
    var index = 0, inputIndex = 0;
    while (index < length){
        result[index++] = inputL[inputIndex];
        result[index++] = inputR[inputIndex];
        inputIndex++;
    }
    return result;
}

function floatTo16BitPCM(output, offset, input){
for (var i = 0; i < input.length; i++, offset += 2){
    var s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
}
}

function writeString(view, offset, string){
for (var i = 0; i < string.length; i++){
    view.setUint8(offset + i, string.charCodeAt(i));
}
}

function encodeWAV(samples){
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);

  /* RIFF identifier */
    writeString(view, 0, 'RIFF');
  /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
    writeString(view, 8, 'WAVE');
  /* format chunk identifier */
    writeString(view, 12, 'fmt ');
  /* format chunk length */
    view.setUint32(16, 16, true);
  /* sample format (PCM) */
    view.setUint16(20, 1, true);
  /* channel count */
    view.setUint16(22, numChannels, true);
  /* sample rate */
    view.setUint32(24, sampleRate, true);
  /* byte rate (sampleRate * blockAlign) */
    view.setUint32(28, sampleRate * numChannels * 2, true);
  /* block align (channel count * bytes per sample) */
    view.setUint16(32, numChannels * 2, true);
  /* bits per sample */
    view.setUint16(34, 16, true);
  /* data chunk identifier */
    writeString(view, 36, 'data');
  /* data chunk length */
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return view;
}

function clear(){
    initBuffers();
}
