const socket = new WebSocket(`wss://${window.location.host}/stream`);
socket.binaryType = "arraybuffer"
const canvas = document.getElementById("video");
const sfps = document.getElementById("data-sfps");
const cfps = document.getElementById("data-cfps");
const ctx = canvas.getContext("2d");

const width = canvas.width;
const height = canvas.height;

socket.onopen = () => {
    console.log("established connection");
}

let lastMessage = Date.now();
let sendMessages = false;

const image = new Image();

let lastSrc;

image.onload = () => {
    if (lastSrc) URL.revokeObjectURL(lastSrc);
    ctx.drawImage(image, 0, 0, width, height);
}

socket.onmessage = (event) => {
    const view = new DataView(event.data);
    const fps = view.getUint16(0, true);
    sfps.innerText = `Real FPS: ${fps}fps`;
    cfps.innerText = `Client FPS: ${Math.floor(1 / ((Date.now()-lastMessage) / 1000))}fps`;
    const blob = new Blob([event.data.slice(2)], { type: "image/webp" });
    image.src = URL.createObjectURL(blob);
    lastSrc = image.src;
    lastMessage = Date.now();
}

socket.onopen = () => {
    sendMessages = true;
}

let allowInput = false;
let input = {
    mouse_buttons: [false, false, false],
    mouse_movement: [0, 0],
    keys: {}
}

const keys_array = ["K_e", "K_ESCAPE", "K_1", "K_2", "K_3", "K_4", "K_5", "K_6", "K_7", "K_8", "K_9", "K_w", "K_a", "K_s", "K_d", "K_SPACE", "K_LSHIFT", "K_RSHIFT", "K_LCTRL", "K_RCTRL", "K_q"];
keys_array.forEach(k => { input.keys[k] = false });
const keyMap = {
    KeyE: "K_e",
    KeyW: "K_w",
    KeyA: "K_a",
    KeyS: "K_s",
    KeyD: "K_d",
    KeyQ: "K_q",
    Digit1: "K_1",
    Digit2: "K_2",
    Digit3: "K_3",
    Digit4: "K_4",
    Digit5: "K_5",
    Digit6: "K_6",
    Digit7: "K_7",
    Digit8: "K_8",
    Digit9: "K_9",
    Escape: "K_ESCAPE",
    Space: "K_SPACE",
    ControlLeft: "K_LCTRL",
    ControlRight: "K_RCTRL",
    ShiftLeft: "K_LSHIFT",
    ShiftRight: "K_RSHIFT"
}

canvas.addEventListener("click", async () => {
    await canvas.requestPointerLock({
        unadjustedMovement: true
    })
})

document.addEventListener("pointerlockchange", (ple) => {
    if (document.pointerLockElement == canvas) {
        allowInput = true;
        console.log("grabbed");
    } else {
        allowInput = false;
        console.log("ungrabbed")
    }
})

let mousePosUpdated = false;

document.addEventListener("mousemove", (me) => {
    if (!allowInput) return;
    input.mouse_movement = [me.movementY, me.movementX];
    mousePosUpdated = true;
})

function updateMouseButtonState(event) {
    const mouseButtonState = [false, false, false];
    mouseButtonState[0] = (event.buttons & 1) !== 0;
    mouseButtonState[1] = (event.buttons & 4) !== 0;
    mouseButtonState[2] = (event.buttons & 2) !== 0;
    input.mouse_buttons = mouseButtonState;
}

document.addEventListener("mousedown", (me) => {
    if (!allowInput) return;
    updateMouseButtonState(me);
})

document.addEventListener("mouseup", (me) => {
    if (!allowInput) return;
    updateMouseButtonState(me);
})

document.addEventListener("keydown", (ke) => {
    if (!allowInput) return;
    if (ke.code in keyMap) {
        input.keys[keyMap[ke.code]] = true;
    }
})

document.addEventListener("keyup", (ke) => {
    if (!allowInput) return;
    if (ke.code in keyMap) {
        input.keys[keyMap[ke.code]] = false;
    }
})

function arraysEqual(a, b) {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length !== b.length) return false;
  
    for (var i = 0; i < a.length; ++i) {
      if (a[i] !== b[i]) return false;
    }
    return true;
}

setInterval(() => {
    if ( mousePosUpdated === false ) {
        input.mouse_movement = [0, 0];
    }
    if (sendMessages) socket.send(JSON.stringify({ input }));
    mousePosUpdated = false;
}, 1000/20);

const radios = document.querySelectorAll("input[type=\"radio\"][name=\"aspect\"]");

setInterval(() => {
    radios.forEach(radio => {
        if (radio.checked) {
            if (radio.value == "16x9") {
                canvas.style = "aspect-ratio: 16 / 9;";
            } else if (radio.value == "4x3") {
                canvas.style = "aspect-ratio: 4 / 3;";
            } else {
                canvas.style = "aspect-ratio: 1;";
            }
        }
    })
}, 1000/10);