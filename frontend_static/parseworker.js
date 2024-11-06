self.onmessage = function(event) {
    const { data, width, height } = event.data;

    // Perform parsing using the optimized function
    const rgbaData = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
        const scanline = new Uint8ClampedArray(width * 4);
        for (let scanIndex = 0; scanIndex < width; scanIndex++) {
            let ri = (y * width) + scanIndex;
            let gi = ri + width;
            let bi = gi + width;
            scanline[scanIndex*4] = data[ri];
            scanline[scanIndex*4+1] = data[gi];
            scanline[scanIndex*4+2] = data[bi];
            scanline[scanIndex*4+3] = 255;
        }
        rgbaData.set(scanline, y*width);
    }
    

    // Create ImageData from the parsed data
    const imageData = new ImageData(rgbaData, width, height);

    // Send the ImageData back to the main thread
    self.postMessage(imageData);
};
