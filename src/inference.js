const tfjs = require('@tensorflow/tfjs-node');

// load model neural network
const loadModel = () => {
  const modelUrl = 'file://ml-model/model.json';
  return tfjs.loadLayersModel(modelUrl);
};


// prediksi data dari model yang telah di load
const predict = (model, imageBuffer) => {
  const tensor = tfjs.node
    .decodeJpeg(imageBuffer) // untuk melakukan decoding dari gambar dengan format JPEG yg disimpan dalam buffer.
    .resizeNearestNeighbor([150, 150]) // ubah gambar dari decode -> 150 x 150 px with "Nearest Neighbor"
    .expandDims() // + ekstra dimensi
    .toFloat(); // konversi -> float

  return model.predict(tensor).data();
}

module.exports = {
  loadModel,
  predict
}