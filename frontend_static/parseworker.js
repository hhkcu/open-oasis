self.onmessage = function(event) {
    const { data, width, height } = event.data;

    // Perform parsing using the optimized function
    const rgbaData = new Uint8ClampedArray(width * height * 4);
    let dataIndex = 0;

    for (let i = 0; i < width * height; i++) {
        rgbaData[i * 4] = data[dataIndex++];       // Red
        rgbaData[i * 4 + 1] = data[dataIndex++];   // Green
        rgbaData[i * 4 + 2] = data[dataIndex++];   // Blue
        rgbaData[i * 4 + 3] = 255;                 // Alpha (fully opaque)
    }

    // Create ImageData from the parsed data
    const imageData = new ImageData(rgbaData, width, height);

    // Send the ImageData back to the main thread
    self.postMessage(imageData);
};
