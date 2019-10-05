"use strict";

/**
 * Check if a sample contains any missing value ("?")
 * @param {Object} object
 * @returns {Boolean}
 */
function areAllValuesValid(object){
  let areValid = true;

  for (const [key, value] of Object.entries(object)) {
    if (value === "?") areValid = false;
  }
  return areValid;
};

/**
 * Remove bad samples (missing values) from a dataset (Array)
 * @param {Array} dataset
 * @returns {Array}
 */
function removeBadSamples(dataset) {
  return dataset.filter(object => areAllValuesValid(object));
}

/**
 * Convert dataset string values to Float
 * @param {Array} dataset 
 */
function datasetToFloat(dataset) {
  dataset.forEach(sample => {
    for (let [key, value] of Object.entries(sample)) {
      sample[key] = parseFloat(value);
    }
  });
  return dataset;
}
/**
 * One-hot encode labes in the dataset
 * @param {Array} dataset 
 */
function oneHotEncodeOrigin(dataset) {
  dataset.forEach(sample => {
    if (sample.origin === 1) {
      sample.USA = 1;
      sample.Europe = 0;
      sample.Japan = 0;
    } else if (sample.origin === 2) {
      sample.USA = 0;
      sample.Europe = 1;
      sample.Japan = 0;
    } else {
      sample.USA = 0;
      sample.Europe = 0;
      sample.Japan = 1;
    }
    delete sample.origin;
  });
  return dataset;
}

export  {
  removeBadSamples,
  datasetToFloat,
  oneHotEncodeOrigin
};
