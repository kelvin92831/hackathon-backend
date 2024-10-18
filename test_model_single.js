const fs = require('fs');
const sharp = require('sharp');
const ort = require('onnxruntime-web');

let session;
(async () => {
  session = await ort.InferenceSession.create('./clip_model.onnx');
  console.log('ONNX 模型加載成功');

  const realimagePath = ['./whichfaceisreal/0_real/10286.jpeg', './whichfaceisreal/0_real/15308.jpeg', './whichfaceisreal/0_real/26383.jpeg', './whichfaceisreal/0_real/28445.jpeg', './whichfaceisreal/0_real/33378.jpeg', './whichfaceisreal/0_real/33926.jpeg', 'whichfaceisreal/0_real/36109.jpeg', './whichfaceisreal/0_real/45244.jpeg', './whichfaceisreal/0_real/56166.jpeg', 'whichfaceisreal/0_real/59841.jpeg']; // 替換成你的測試圖片路徑
  const fakeimagePath = ['./whichfaceisreal/1_fake/image-2019-02-17_033657.jpeg', './whichfaceisreal/1_fake/image-2019-02-17_071948.jpeg', './whichfaceisreal/1_fake/image-2019-02-17_151253.jpeg', './whichfaceisreal/1_fake/image-2019-02-18_005909.jpeg', './whichfaceisreal/1_fake/image-2019-02-18_041109.jpeg', './whichfaceisreal/1_fake/image-2019-02-18_045610.jpeg', './whichfaceisreal/1_fake/image-2019-02-18_060447.jpeg', './whichfaceisreal/1_fake/image-2019-02-18_120927.jpeg']; // 替換成你的測試圖片路徑
  
  //const imageBuffer = fs.readFileSync(realimagePath[0]);
  const imageBuffer = fs.readFileSync(fakeimagePath[1]);


  const processedImage = await preprocessImage(imageBuffer);

  const input = tensorToONNXInput(processedImage);

  const feeds = { 'x': input }; 

  const results = await session.run(feeds);

  const outputTensor = results['4157'];
  const prediction = outputTensor.data[0];

  const sigmoid = (x) => 1 / (1 + Math.exp(-x)); 
  const finalPrediction = sigmoid(prediction); 

  const predictType = (finalPrediction < 0.01) ? 'Real' : 'Fake';

  console.log(outputTensor);  
  console.log(finalPrediction);
  console.log(predictType);
})();

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

const tensorToONNXInput = (tensor) => {
  return tensor;
};
