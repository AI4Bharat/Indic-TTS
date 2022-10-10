/* Audio recording and streaming demo by Miguel Grinberg.

   Adapted from https://webaudiodemos.appspot.com/AudioRecorder
   Copyright 2013 Chris Wilson

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

//Create instance of streaming client.

// var socket_speech= io('wss://asr-api.ai4bharat.org/speech',{
//     'path': '/tts_socket.io',
//     'transport': ['websocket'],
//     'upgrade':false
//     });
// var socket_speech= io('ws://127.0.0.1:5000/tts',{
//     'path': '/tts_socket.io',
//     'transport': ['websocket'],
//     'upgrade':false
// });
// socket_speech.once("connect", (x) => {
//     console.log("reached")
//     console.log(x)
//     console.log(socket_speech.id); // "G5p5..."
// });

// const SOCKET_URL = 'ws://0.0.0.0:5001/tts'
// const SOCKET_URL = 'ws://216.48.183.5:5001/tts'
const SOCKET_URL = 'wss://tts-api.ai4bharat.org/tts'
var socket_tts= io(SOCKET_URL, {
    'path': '/tts_socket.io',
    'transport': ['websocket'],
    'upgrade':false
    });
// var socket_tts= io('ws://127.0.0.1:5000/text',transport=['websocket'],upgrade=false,path='/tts_socket.io');

socket_tts.once("connect", (x) => {
    console.log(x)
    console.log(socket_tts.id); // "G5p5..."
});

window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext({sampleRate: 16000});
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;

// var socketio = io.connect('wss://216.48.182.174:4998/') //, {transports: ['websocket']});//'http://127.0.0.1:5001') //, {transports: ['websocket']});
// var socketio = io.connect('wss://216.48.182.174:4990',{
//     'reconnect': false,
//     'reconnection delay': 1000,
//     'max reconnection attempts': 5
// }); //, {transports: ['websocket']});//'http://127.0.0.1:5001') //, {transports: ['websocket']});
// socketio.emit('connection','Connected',function(z) {
//     console.log(z)
// });

// socketio.on('output',(op)=>{
//      document.getElementById('text').innerHTML += op + ' ' 
// });

function hindi_select(){
    console.log("Hindi button clicked.");
    socketio.emit('lang_select','hi');
    // var en_button = document.getElementById("en_button");
    // en_button.classList.remove("is-orange");
    var hi_button = document.getElementById("hi_button");
    hi_button.classList.add("is-orange");
    var mr_button = document.getElementById("mr_button");
    mr_button.classList.remove("is-orange");
}

// function english_select(){
//     console.log("English button clicked.");
//     socketio.emit('lang_select','en');
//     // var en_button = document.getElementById("en_button");
//     // en_button.classList.add("is-orange");
//     var mr_button = document.getElementById("mr_button");
//     mr_button.classList.remove("is-orange");
//     var hi_button = document.getElementById("hi_button");
//     hi_button.classList.remove("is-orange");
// }

function marathi_select(){
    console.log("Marathi button clicked.");
    socketio.emit('lang_select','mr');
    // var en_button = document.getElementById("en_button");
    // en_button.classList.remove("is-orange");
    var hi_button = document.getElementById("hi_button");
    hi_button.classList.remove("is-orange");
    var mr_button = document.getElementById("mr_button");
    mr_button.classList.add("is-orange");
}

function stream_select(){
    console.log("Hindi button clicked.");
    var hi_button = document.getElementById("speech");
    hi_button.style.display = 'block';
    var mr_button = document.getElementById("app");
    mr_button.style.display = 'none';

    var hi_button = document.getElementById("streaming");
    hi_button.classList.add("is-orange");
    var mr_button = document.getElementById("command");
    mr_button.classList.remove("is-orange");
}

function command_select(){
    console.log("Marathi button clicked.");
    var hi_button = document.getElementById("speech");
    hi_button.style.display = 'none';
    var mr_button = document.getElementById("app");
    mr_button.style.display = 'block';

    var hi_button = document.getElementById("streaming");
    hi_button.classList.remove("is-orange");
    var mr_button = document.getElementById("command");
    mr_button.classList.add("is-orange");
}

function chngimg() {
    console.log("image clicked")
    var img = document.getElementById('record').src;
    if (img.indexOf('mic.gif')!=-1) {
        document.getElementById('record').src  = 'https://gist.githubusercontent.com/bietkul/20f702276adff150f3cc4502254665d2/raw/02a339636df69878b48608468f4f25333d3ef8c9/animation.gif';
    }
     else {
       document.getElementById('record').src = 'https://gist.githubusercontent.com/bietkul/20f702276adff150f3cc4502254665d2/raw/02a339636df69878b48608468f4f25333d3ef8c9/mic.gif';
   }

}


function toggleRecording( e ) {

    console.log("image clicked")
    var img = document.getElementById('record').src;
    if (img.indexOf('mic.gif')!=-1) {
        document.getElementById('record').src  = 'https://gist.githubusercontent.com/bietkul/20f702276adff150f3cc4502254665d2/raw/02a339636df69878b48608468f4f25333d3ef8c9/animation.gif';
    }
    else {
       document.getElementById('record').src = 'https://gist.githubusercontent.com/bietkul/20f702276adff150f3cc4502254665d2/raw/02a339636df69878b48608468f4f25333d3ef8c9/mic.gif';
    }

    audioContext.resume()
    if (e.classList.contains('recording')) {
        e.classList.remove('recording');
        recording = false;
        socketio.emit('end_recording',"ended");
        // document.getElementById('text').innerHTML = ''

    } else {
        // start recording

        e.classList.add('recording');
        recording = true;
        document.getElementById('text').innerHTML = ''
        // socketio.emit('rec', 'hindi');
    }
}

function convertToMono( input ) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect( splitter );
    splitter.connect( merger, 0, 0 );
    splitter.connect( merger, 0, 1 );
    return merger;
}

// function cancelAnalyserUpdates() {
//     window.cancelAnimationFrame( rafID );
//     rafID = null;
// }

// function updateAnalysers(time) {
//     if (!analyserContext) {
//         var canvas = document.getElementById('analyser');
//         canvasWidth = canvas.width;
//         canvasHeight = canvas.height;
//         analyserContext = canvas.getContext('2d');
//     }

//     // analyzer draw code here
//     {
//         var SPACING = 3;
//         var BAR_WIDTH = 1;
//         var numBars = Math.round(canvasWidth / SPACING);
//         var freqByteData = new Uint8Array(analyserNode.frequencyBinCount);

//         analyserNode.getByteFrequencyData(freqByteData); 

//         analyserContext.clearRect(0, 0, canvasWidth, canvasHeight);
//         analyserContext.fillStyle = '#F6D565';
//         analyserContext.lineCap = 'round';
//         var multiplier = analyserNode.frequencyBinCount / numBars;

//         // Draw rectangle for each frequency bin.
//         for (var i = 0; i < numBars; ++i) {
//             var magnitude = 0;
//             var offset = Math.floor( i * multiplier );
//             // gotta sum/average the block, or we miss narrow-bandwidth spikes
//             for (var j = 0; j< multiplier; j++)
//                 magnitude += freqByteData[offset + j];
//             magnitude = magnitude / multiplier;
//             var magnitude2 = freqByteData[i * multiplier];
//             analyserContext.fillStyle = "hsl( " + Math.round((i*360)/numBars) + ", 100%, 50%)";
//             analyserContext.fillRect(i * SPACING, canvasHeight, BAR_WIDTH, -magnitude);
//         }
//     }
    
//     // rafID = window.requestAnimationFrame( updateAnalysers );
// }

function toggleMono() {
    if (audioInput != realAudioInput) {
        audioInput.disconnect();
        realAudioInput.disconnect();
        audioInput = realAudioInput;
    } else {
        realAudioInput.disconnect();
        audioInput = convertToMono( realAudioInput );
    }

    audioInput.connect(inputPoint);
}

function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;

    audioInput = convertToMono( audioInput );
    audioInput.connect(inputPoint);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect( analyserNode );

    scriptNode = (audioContext.createScriptProcessor || audioContext.createJavaScriptNode).call(audioContext, 1024, 1, 1);
    scriptNode.onaudioprocess = function (audioEvent) {
        if (recording) {
            input = audioEvent.inputBuffer.getChannelData(0);

            // convert float audio data to 16-bit PCM
            var buffer = new ArrayBuffer(input.length * 2)
            var output = new DataView(buffer);
            for (var i = 0, offset = 0; i < input.length; i++, offset += 2) {
                var s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            socketio.emit('audio', buffer)
        
        }
    }
    inputPoint.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect( zeroGain );
    zeroGain.connect( audioContext.destination );
    // updateAnalysers();
}

// function initAudio() {
    
//     if (!navigator.getUserMedia)
//         navigator.getUserMedia = navigator.getUserMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
//     if (!navigator.cancelAnimationFrame)
//         navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
//     if (!navigator.requestAnimationFrame)
//         navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;


//     navigator.getUserMedia({audio: true}, gotStream, function(e) {
//         // alert('Error getting audio');
//         console.log('error getting audio')
//         console.log(e);
//     });
// }

// window.addEventListener('load', initAudio );