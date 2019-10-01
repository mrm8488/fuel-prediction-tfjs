"use strict";

import "babel-polyfill";

import * as tf from "@tensorflow/tfjs";

import trainingSet from "./mpg.json";

import {
  removeBadSamples,
  datasetToFloat,
  oneHotEncodeOrigin
} from "./preprocess";

import normalize from "./stats";
import { stat } from "fs";

let predictButton = document.getElementsByClassName("predict")[0];
let isTrainingMsg = document.getElementById("isTraining");

const stats = {
  mean: {
    cylinders: 0,
    displacement: 0,
    horsepower: 0,
    weight: 0,
    acceleration: 0,
    model_year: 0
  },
  stdev: {
    cylinders: 0,
    displacement: 0,
    horsepower: 0,
    weight: 0,
    acceleration: 0,
    model_year: 0
  }
};

let trainDataTensor, testDataTensor, trainLabelsTensor, testLabelsTensor, model;

let training = true;

const prepareData = trainingSet => {
  console.table(trainingSet[0]);
  const totalTraingSamples = trainingSet.length;
  console.log(`total datos cargados: ${totalTraingSamples}`);

  const cleanedDataset = oneHotEncodeOrigin(
    datasetToFloat(removeBadSamples(trainingSet))
  );
  console.log(
    `datos después de eliminar valores inválidos, pasar a Number y one-hot encode origin: `
  );
  console.log(cleanedDataset.length);
  console.log(cleanedDataset[0]);

  const shuffle = array => {
    let j, x, i;
    for (i = array.length - 1; i > 0; i--) {
      j = Math.floor(Math.random() * (i + 1));
      x = array[i];
      array[i] = array[j];
      array[j] = x;
    }
    return array;
  };

  const numItemsTraining = ~~(0.8 * cleanedDataset.length);
  console.log(`datos para entrenamiento 80%: ${numItemsTraining}`);

  const shuffledDataSet = shuffle(cleanedDataset);
  console.log(`muestra de dato despues de shuffle: `);
  console.log(shuffledDataSet[0]);

  let trainDataSet = [];
  for (let i = 0; i < numItemsTraining; i++) {
    trainDataSet.push(shuffledDataSet[i]);
  }

  console.log(`Tamaño final trainDataSet: ${trainDataSet.length}`);

  let testDataSet = [];
  for (let i = numItemsTraining; i < shuffledDataSet.length; i++) {
    testDataSet.push(shuffledDataSet[i]);
  }

  console.log(`Tamaño final testDataSet: ${testDataSet.length}`);

  console.assert(
    testDataSet.length + trainDataSet.length === cleanedDataset.length,
    "Los datasets tienen difrente tamaño"
  );

  const getLabels = dataset => {
    const labels = dataset.map(sample => +sample.mpg);
    return labels;
  };

  const trainLabels = getLabels(trainDataSet);
  const testLabels = getLabels(testDataSet);

  const normalizeDataset = dataset => {
    dataset.forEach(sample => {
      for (let [key, value] of Object.entries(sample)) {
        if (
          key !== "USA" &&
          key !== "Europe" &&
          key !== "Japan" &&
          key !== "mpg"
        ) {
          let result = normalize(value, dataset.map(sample => sample[key]));
          sample[key] = result.normalizedValue;
          stats.mean[key] = result.mean;
          stats.stdev[key] = result.stdDev;
        }
        delete sample["mpg"];
      }
    });
    return dataset;
  };

  const normTrainData = normalizeDataset(trainDataSet);
  const normTestData = normalizeDataset(testDataSet);

  console.log(normTrainData.length);
  console.table(stats);

  trainDataTensor = tf.tensor2d(
    normTrainData.map(sample => [
      sample.cylinders,
      sample.displacement,
      sample.horsepower,
      sample.weight,
      sample.acceleration,
      sample.model_year,
      sample.USA,
      sample.Europe,
      sample.Japan
    ]),
    [normTrainData.length, 9]
  );

  testDataTensor = tf.tensor2d(
    normTestData.map(sample => [
      sample.cylinders,
      sample.displacement,
      sample.horsepower,
      sample.weight,
      sample.acceleration,
      sample.model_year,
      sample.USA,
      sample.Europe,
      sample.Japan
    ]),
    [normTestData.length, 9]
  );

  //console.log(trainLabels.length);
  //console.log(testLabels.length);
  trainLabelsTensor = tf.tensor2d(trainLabels, [trainLabels.length, 1]);
  testLabelsTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);
};

const buildModel = () => {
  model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: 9,
      activation: "relu",
      units: 64
    })
  );

  model.add(
    tf.layers.dense({
      inputShape: 64,
      units: 64,
      activation: "relu"
    })
  );

  model.add(
    tf.layers.dense({
      inputShape: 64,
      units: 1
    })
  );

  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.rmsprop(0.001),
    metrics: ["mae"]
  });
};

const trainData = async () => {
  let numSteps = 10;
  let trainingStepsDiv = document.getElementsByClassName("training-steps")[0];
  for (let i = 0; i < numSteps; i++) {
    let res = await model.fit(trainDataTensor, trainLabelsTensor, {
      epochs: 100
    });
    console.log(
      `Training step: ${i}/${numSteps - 1}, loss: ${res.history.loss[0]}`
    );
    console.log(`MAE: ${res.history.mae[0]}`);
    trainingStepsDiv.innerHTML = `Training step: ${i}/${numSteps -
      1}, loss: ${res.history.loss[0].toFixed(
      3
    )}, MAE: ${res.history.mae[0].toFixed(3)}`;
    if (i === numSteps - 1) training = false;
  }
};

//trainData().catch(console.error);

const getInputData = () => {
  console.log("entra a get input data");
  let cilynders = document.getElementsByName("cilynders")[0].value;
  let displacement = document.getElementsByName("displacement")[0].value;
  let horsepower = document.getElementsByName("horsepower")[0].value;
  let weight = document.getElementsByName("weight")[0].value;
  let acceleration = document.getElementsByName("acceleration")[0].value;
  let model_year = document.getElementsByName("model_year")[0].value;
  let e = document.getElementById("country");
  let country = e.options[e.selectedIndex].value.split(" ");
  let USA = country[0];
  let Europe = country[1];
  let Japan = country[2];

  const inputData = {
    cylinders: Number(
      (cilynders - stats.mean.cylinders) / stats.stdev.cylinders
    ),
    displacement: Number(
      (displacement - stats.mean.displacement) / stats.stdev.displacement
    ),
    horsepower: Number(
      (horsepower - stats.mean.horsepower) / stats.stdev.horsepower
    ),
    weight: Number((weight - stats.mean.weight) / stats.stdev.weight),
    acceleration: Number(
      (acceleration - stats.mean.acceleration) / stats.stdev.acceleration
    ),
    model_year: Number(
      (model_year - stats.mean.model_year) / stats.stdev.model_year
    ),
    USA: Number(USA),
    Europe: Number(Europe),
    Japan: Number(Japan)
  };
  return inputData;
};

const predict = async inputData => {

  console.log("Entra a predict con los siguientes valores:");
  console.table(inputData);

  let newDataTensor = tf.tensor2d(
    [inputData].map(item => [
      item.cylinders,
      item.displacement,
      item.horsepower,
      item.weight,
      item.acceleration,
      item.model_year,
      item.USA,
      item.Europe,
      item.Japan
    ]),
    [1, 9]
  );

  let prediction = model.predict(newDataTensor);

  displayPrediction(prediction);
};

const displayPrediction = prediction => {
  console.log(prediction);
  let predictionDiv = document.getElementsByClassName("prediction")[0];
  let predictionSection = document.getElementsByClassName(
    "prediction-block"
  )[0];

  predictionDiv.innerHTML = prediction;
  predictionSection.style.display = "block";
};

const init = async () => {
  prepareData(trainingSet);
  buildModel();
  isTrainingMsg.style.display = "block";
  await trainData();
  if (!training) {
    isTrainingMsg.style.display = "none";
    predictButton.disabled = false;
    predictButton.onclick = () => {
      const inputData = getInputData();
      predict(inputData);
    };
  }
};

init().catch(console.error);
