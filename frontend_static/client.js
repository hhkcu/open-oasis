const socket = new WebSocket(`ws://${window.location.host}`);
socket.binaryType = "arraybuffer"
const canvas = document.getElementById("video");
const ctx = canvas.getContext("2d");

const width = canvas.width;
const height = canvas.height;

const worker = new Worker("/s/parseworker.js");

socket.onopen = () => {
    console.log("established connection");
}

socket.onmessage = (event) => {
    const fps = (new Uint32Array(event.data))[0]
}

worker.onmessage = (event) => {
    const imageData = event.data;
    ctx.putImageData(imageData, 0, 0)
}