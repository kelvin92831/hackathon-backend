const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const sharp = require('sharp');
const ort = require('onnxruntime-web');

const app = express();
app.use(bodyParser.json({ limit: '50mb' })); 

let session;

// load onnx model
(async () => {
  session = await ort.InferenceSession.create('./clip_model.onnx');
  console.log('ONNX 模型加載成功');
})();

//get request
app.post('/predict', async (req, res) => {
    try {
      const { image } = req.body;
  
      if (!image) {
        return res.status(400).json({ error: 'No image data provided' });
      }
  
      // Base64 to Buffer
      const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
      const imageBuffer = Buffer.from(base64Data, 'base64');
  
      // preprocessing
      const processedImage = await preprocessImage(imageBuffer);
  
      // to model
      const input = tensorToONNXInput(processedImage);
      const feeds = { 'x': input }; 
      const results = await session.run(feeds);
      
      const outputTensor = results['4157']; 
      const prediction = outputTensor.data[0];
  
      const sigmoid = (x) => 1 / (1 + Math.exp(-x));
      const finalPrediction = sigmoid(prediction);
  
      const predictType = (finalPrediction < 0.5) ? 'Real' : 'Fake';
  
      res.json({
        success: true,
        prediction: finalPrediction,
        type: predictType
      });
  
    } catch (error) {
      console.error('Error during prediction:', error);
      res.status(500).json({ error: 'Internal Server Error' });
    }
  });
  
  // server listening
  app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
  });

  const preprocessImage = async (imgBuffer) => {
    const resizedImage = await sharp(imgBuffer)
      .resize(224, 224, {
        fit: 'cover',
        kernel: sharp.kernel.cubic 
      })
      .toFormat('png') 
      .raw()
      .toBuffer({ resolveWithObject: true });
  
    const imgData = new Float32Array(resizedImage.data).map(val => val / 255.0);
  
    const imageTensor = new ort.Tensor('float32', imgData, [1, 224, 224, 3]);
  
    const nchwTensor = transposeNHWCtoNCHW(imageTensor);
  
    const mean = [0.48145466, 0.4578275, 0.40821073];
    const std = [0.26862954, 0.26130258, 0.27577711];
  
    for (let i = 0; i < 3 * 224 * 224; i += 1) {
      nchwTensor.data[i] = (nchwTensor.data[i] - mean[Math.floor(i / (224 * 224))]) / std[Math.floor(i / (224 * 224))];
    }
  
    return nchwTensor;
  };
  
  const transposeNHWCtoNCHW = (tensor) => {
    const [batch, height, width, channels] = tensor.dims; // NHWC 形狀
    const nchwData = new Float32Array(batch * channels * height * width); // NCHW 形狀
    let idx = 0;
  
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            nchwData[idx++] = tensor.data[b * height * width * channels + h * width * channels + w * channels + c];
          }
        }
      }
    }
  
    return new ort.Tensor('float32', nchwData, [batch, channels, height, width]);
  };
  
  // onnx to tensor
  const tensorToONNXInput = (tensor) => {
    return tensor;
  };
