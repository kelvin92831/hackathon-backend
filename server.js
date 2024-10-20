// const express = require('express');
// const bodyParser = require('body-parser');
// const fs = require('fs');
// const sharp = require('sharp');
// const ort = require('onnxruntime-web');

// const app = express();
// app.use(bodyParser.json({ limit: '50mb' })); 

// let session;

// // load onnx model
// (async () => {
//   session = await ort.InferenceSession.create('./clip_model.onnx');
//   console.log('ONNX 模型加載成功');
// })();

// //get request
// app.post('/predict', async (req, res) => {
//     try {
//       const { image } = req.body;
  
//       if (!image) {
//         return res.status(400).json({ error: 'No image data provided' });
//       }
  
//       // Base64 to Buffer
//       const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
//       const imageBuffer = Buffer.from(base64Data, 'base64');
  
//       // preprocessing
//       const processedImage = await preprocessImage(imageBuffer);
  
//       // to model
//       const input = tensorToONNXInput(processedImage);
//       const feeds = { 'x': input }; 
//       const results = await session.run(feeds);
      
//       const outputTensor = results['4157']; 
//       const prediction = outputTensor.data[0];
  
//       const sigmoid = (x) => 1 / (1 + Math.exp(-x));
//       const finalPrediction = sigmoid(prediction);
  
//       const predictType = (finalPrediction < 0.5) ? 'Real' : 'Fake';
  
//       res.json({
//         success: true,
//         prediction: finalPrediction,
//         type: predictType
//       });
  
//     } catch (error) {
//       console.error('Error during prediction:', error);
//       res.status(500).json({ error: 'Internal Server Error' });
//     }
//   });
  
//   // server listening
//   app.listen(3000, () => {
//     console.log('Server running on http://localhost:3000');
//   });

//   const preprocessImage = async (imgBuffer) => {
//     const resizedImage = await sharp(imgBuffer)
//       .resize(224, 224, {
//         fit: 'cover',
//         kernel: sharp.kernel.cubic 
//       })
//       .toFormat('png') 
//       .raw()
//       .toBuffer({ resolveWithObject: true });
  
//     const imgData = new Float32Array(resizedImage.data).map(val => val / 255.0);
  
//     const imageTensor = new ort.Tensor('float32', imgData, [1, 224, 224, 3]);
  
//     const nchwTensor = transposeNHWCtoNCHW(imageTensor);
  
//     const mean = [0.48145466, 0.4578275, 0.40821073];
//     const std = [0.26862954, 0.26130258, 0.27577711];
  
//     for (let i = 0; i < 3 * 224 * 224; i += 1) {
//       nchwTensor.data[i] = (nchwTensor.data[i] - mean[Math.floor(i / (224 * 224))]) / std[Math.floor(i / (224 * 224))];
//     }
  
//     return nchwTensor;
//   };
  
//   const transposeNHWCtoNCHW = (tensor) => {
//     const [batch, height, width, channels] = tensor.dims; // NHWC 形狀
//     const nchwData = new Float32Array(batch * channels * height * width); // NCHW 形狀
//     let idx = 0;
  
//     for (let b = 0; b < batch; b++) {
//       for (let c = 0; c < channels; c++) {
//         for (let h = 0; h < height; h++) {
//           for (let w = 0; w < width; w++) {
//             nchwData[idx++] = tensor.data[b * height * width * channels + h * width * channels + w * channels + c];
//           }
//         }
//       }
//     }
  
//     return new ort.Tensor('float32', nchwData, [batch, channels, height, width]);
//   };
  
//   // onnx to tensor
//   const tensorToONNXInput = (tensor) => {
//     return tensor;
//   };


//-----------------------------
// const express = require("express");
// const multer = require("multer");
// const ort = require("onnxruntime-node");
// const path = require("path");
// const ffmpeg = require('fluent-ffmpeg');
// const fs = require("fs");
// const sharp = require('sharp');

// const app = express();
// const PORT = 3000;

// // Set up multer for handling file uploads
// const upload = multer({ dest: "uploads/" });

// // Load ONNX models
// let baseSession;
// let modelSession;

// ort.InferenceSession.create("./models/Xception.onnx").then((s) => {
//   baseSession = s;
// });

// ort.InferenceSession.create("./models/Xception_video2.onnx").then((s) => {
//   modelSession = s;
// });

// // Preprocess input for ONNX model
// function preprocessFrame(frameBuffer) {
//   // Preprocess the frame (normalize, etc.)
//   const inputTensor = new Float32Array(frameBuffer); // Adjust for preprocessing logic
//   return inputTensor;
// }

// function extractFramesAndProcess(videoPath, baseModel, frameRate) {
//   return new Promise((resolve, reject) => {
//     const frameFolder = path.join(__dirname, "frames");  // Save frames in a folder
//     const processedFrames = [];

//     // Ensure the frame directory exists
//     if (!fs.existsSync(frameFolder)) {
//       fs.mkdirSync(frameFolder);
//     }

//     ffmpeg(videoPath)
//       .on("error", (err) => reject(err))
//       .on("end", async () => {
//         try {
//           // Read all images from the folder and process them
//           const files = fs.readdirSync(frameFolder);

//           for (const file of files) {
//             const framePath = path.join(frameFolder, file);
//             const frameBuffer = fs.readFileSync(framePath);  // Load the frame into memory

//             // Optionally use sharp to resize/process the image as a buffer
//             const resizedFrame = await sharp(frameBuffer)
//               .resize({ width: 299, height: 299 })
//               .toBuffer();

//             const preprocessed = preprocessFrame(resizedFrame);
//             const features = await baseModel.run({ input: preprocessed });
//             processedFrames.push(features.output);  // Collect Xception features
//           }

//           resolve(processedFrames);
//         } catch (err) {
//           reject(err);
//         }
//       })
//       .screenshots({
//         count: 60,  // Extract 60 frames from the video
//         folder: frameFolder,
//         size: "299x299",
//       });
//   });
// }

// // Process the video and extract features
// function processVideo(videoPath, frameRate) {
//   return new Promise((resolve, reject) => {
//     const frames = [];
//     const processedFrames = [];

//     // Use ffmpeg to extract frames from the video
//     ffmpeg(videoPath)
//       .on("error", reject)
//       .on("end", async () => {
//         // Process frames with base model (Xception) to extract features
//         for (const frame of frames) {
//           const preprocessed = preprocessFrame(frame);
//           const features = await baseSession.run({ input: preprocessed });
//           processedFrames.push(features.output);
//         }

//         // Ensure that the frame features are padded/truncated to max_frames
//         console.log("Number of frames processed: ", frames.length);
//         resolve(processedFrames);
//       })
//       .screenshots({
//         count: 60, // Number of frames to extract
//         folder: "frames/",
//         size: "299x299",
//       });
//   });
// }

// // Route to handle video upload and processing
// app.post("/upload", upload.single("video"), async (req, res) => {
//   try {
//     const videoPath = req.file.path;
//     console.log("success1");

//     // Extract frames and process them with the base model
//     // const frameFeatures = await processVideo(videoPath, 20);

//     const baseModel = await ort.InferenceSession.create("./models/Xception.onnx");  // Xception in ONNX format
    
//     console.log("success1.5");

//     const frameFeatures = await extractFramesAndProcess(videoPath, baseModel, 20);

//     console.log("success2");

//     // Prepare input tensor for the main model
//     const flattenedFeatures = frameFeatures.flat();  // Flatten the array to match the expected input
//     const dims = [1, 60, 2048];  // Make sure the dimensions are passed as an array
//     const inputTensor = new ort.Tensor("float32", flattenedFeatures, dims);  // Create the tensor with correct dims

//     // const inputTensor = new ort.Tensor("float32", frameFeatures, [1, 60, 2048]);
//     console.log("success3");

//     // Run inference using the main ONNX model
//     const results = await modelSession.run({ input: inputTensor });
//     console.log("success4");

//     // Get the output prediction
//     const predictions = results.output.data;
//     res.json({ predictions });
//     console.log("success5");

//   } catch (err) {
//     console.error(err);
//     res.status(500).send("Error processing the video");
//   }
// });

// // Start the server
// app.listen(PORT, () => {
//   console.log(`Server running on port ${PORT}`);
// });

//-----------------------

const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const ort = require("onnxruntime-node");
const path = require("path");
const fs = require("fs");
const sharp = require('sharp');  // Add sharp to help load and process images

const app = express();
const PORT = 3000;

// Set up multer for handling file uploads
const upload = multer({ dest: "uploads/" });

// Load ONNX model
let session;
ort.InferenceSession.create("./models/Xception_video2.onnx").then((s) => {
  session = s;
});

// Function to preprocess video frames for Xception
function preprocessFrame(frameBuffer) {
  // Convert the frame to float32 and normalize based on Xception's requirements (e.g., divide by 255)
  const inputTensor = new Float32Array(frameBuffer); // Add any additional preprocessing steps here
  return inputTensor;
}

// Function to extract frames from video and pass them to Xception for feature extraction
function extractFramesAndProcess(videoPath, baseModel, frameRate) {
  return new Promise((resolve, reject) => {
    const frameFolder = path.join(__dirname, "frames");  // Folder to store extracted frames
    const processedFrames = [];

    // Ensure the frame directory exists
    if (!fs.existsSync(frameFolder)) {
      fs.mkdirSync(frameFolder);
    }

    ffmpeg(videoPath)
      .on("error", (err) => reject(err))
      .on("end", async () => {
        try {
          // Read all the extracted frame images from the folder
          const files = fs.readdirSync(frameFolder);
          console.log(`Number of frames extracted: ${files.length}`);

          // Check if frames were actually extracted
          if (files.length === 0) {
            throw new Error("No frames were extracted from the video.");
          }

          for (const file of files) {
            const framePath = path.join(frameFolder, file);
            console.log(`Processing frame: ${framePath}`);

            // Load the frame image
            const frameBuffer = fs.readFileSync(framePath);
            if (!frameBuffer) {
              throw new Error(`Error reading frame: ${framePath}`);
            }

            // Resize the frame using sharp (or another method) to 299x299
            const resizedFrame = await sharp(frameBuffer)
              .resize({ width: 299, height: 299 })
              .toBuffer();

            // Preprocess the frame (normalize, etc.)
            const preprocessed = preprocessFrame(resizedFrame);

            // Run the frame through the Xception model to get features
            const features = await baseModel.run({ input: preprocessed });
            processedFrames.push(features.output);  // Store the processed features
          }

          console.log(`Number of frames processed: ${processedFrames.length}`);

          // Resolve with the processed frame features
          resolve(processedFrames);

        } catch (err) {
          reject(err);
        }
      })
      .screenshots({
        count: 60,  // Extract 60 frames from the video
        folder: frameFolder,  // Save them in the frameFolder
        size: "299x299",  // Resize to 299x299 pixels
      });
  });
}


// Route to handle video upload and processing
// app.post("/upload", upload.single("video"), async (req, res) => {
//   try {
//     const videoPath = req.file.path;

//     // Load your ONNX base model (which represents the Xception model)
//     const baseModel = await ort.InferenceSession.create("./models/Xception.onnx");  // Xception in ONNX format

//     console.log("success1");
//     // Extract frames from the video and pass them to Xception
//     const frameFeatures = await extractFramesAndProcess(videoPath, baseModel, 20);

//     console.log("success2");

//     // Prepare the final input tensor for your ONNX model
//     // Ensure that frameFeatures is flattened and the dimensions are passed correctly
//     const flattenedFeatures = frameFeatures.flat();  // Flatten the array to match the expected input
//     const dims = [1, 60, 2048];  // Make sure the dimensions are passed as an array
//     const inputTensor = new ort.Tensor("float32", flattenedFeatures, dims);  // Create the tensor with correct dims
//     console.log("success3");

//     // Run inference using the ONNX model
//     const results = await session.run({ input: inputTensor });
//     console.log("success4");

//     // Get the output prediction
//     const predictions = results.output.data;
//     console.log("success5");

//     res.json({ predictions });
//   } catch (err) {
//     console.error(err);
//     res.status(500).send("Error processing the video");

//   }

// });

app.post("/upload", upload.single("video"), async (req, res) => {
  try {
    const videoPath = req.file.path;

    // Load your ONNX base model (which represents the Xception model)
    const baseModel = await ort.InferenceSession.create("./models/Xception.onnx");  // Xception in ONNX format

    console.log("success1");
    
    // Extract frames from the video and pass them to Xception
    const frameFeatures = await extractFramesAndProcess(videoPath, baseModel, 20);
    console.log("Number of frame features extracted: ", frameFeatures.length);

    // Prepare the final input tensor for your ONNX model
    // Flatten the frame features array and check the size
    const flattenedFeatures = frameFeatures.flat();
    console.log("Flattened features length: ", flattenedFeatures.length);
    console.log("Flattened features (first 10 elements): ", flattenedFeatures.slice(0, 10));
    console.log("Expected tensor size: 122880");

    if (flattenedFeatures.length !== 122880) {
      throw new Error("Mismatch between flattened features length and expected tensor size");
    }

    // Create the input tensor with correct dimensions
    const dims = [1, 60, 2048];
    const inputTensor = new ort.Tensor("float32", flattenedFeatures, dims);
    console.log("Tensor created successfully");

    // Run inference using the ONNX model
    const results = await session.run({ input: inputTensor });
    console.log("Inference completed successfully");

    // Get the output prediction
    const predictions = results.output.data;
    console.log("Prediction result: ", predictions);

    res.json({ predictions });
  } catch (err) {
    console.error(err);
    res.status(500).send("Error processing the video");
  }
});


// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
