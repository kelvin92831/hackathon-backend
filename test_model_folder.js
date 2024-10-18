const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const ort = require('onnxruntime-web');

let session;

(async () => {
  session = await ort.InferenceSession.create('./clip_model.onnx');
  console.log('ONNX 模型加載成功');

  const realImageDir = './Dataset/Test/Real';  
  const fakeImageDir = './Dataset/Test/Fake'; 

  

  console.log('正在測試 Real 資料夾中的圖片...');
  await processImagesInFolder(realImageDir, 'Real');

  console.log('正在測試 Fake 資料夾中的圖片...');
  await processImagesInFolder(fakeImageDir, 'Fake');
})();

const processImagesInFolder = async (folderPath, label) => {
  const files = fs.readdirSync(folderPath); 

  let total = 0;
  let correct = 0;
  let predictionScore = 0

  for (const file of files) {
    if (total == 30) break;
    const filePath = path.join(folderPath, file);

    if (file.endsWith('.jpeg') || file.endsWith('.jpg') || file.endsWith('.png')) {
      console.log(`正在處理圖片: ${file}`);

      const imageBuffer = fs.readFileSync(filePath);
      const processedImage = await preprocessImage(imageBuffer);

      const input = tensorToONNXInput(processedImage);
      const feeds = { 'x': input };
      const results = await session.run(feeds);

      const outputTensor = results['4157'];
      const prediction = outputTensor.data[0];

      const sigmoid = (x) => 1 / (1 + Math.exp(-x)); 
      const finalPrediction = sigmoid(prediction); 

      const predictType = (finalPrediction < 0.05) ? 'Real' : 'Fake'; //調整threshold
      total++;

      if (predictType == label) correct++; 
      console.log(`圖片: ${file}, 預測: ${predictType}, 實際: ${label}, 預測分數: ${finalPrediction}`);
    }
  }
  const accuracy = (correct / total) * 100;
  console.log(`資料夾${folderPath}中圖片總數: ${total}, 正確預測: ${correct}, 正確率: ${accuracy.toFixed(2)}%`);
};


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

// NHWC to NCHW
const transposeNHWCtoNCHW = (tensor) => {
  const [batch, height, width, channels] = tensor.dims;
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


const tensorToONNXInput = (tensor) => {
  return tensor;
};
