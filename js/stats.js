"use strict";

import { mean, std, subtract, divide } from "mathjs";

const calculateMean = column => {
  /*const sum = column.reduce((a, b) => a + b);
  const avg = sum / column.length;*/
  return mean(column);
};

const calculateStdDev = column => {
  /*const avg = calculateMean(column);

  const squareDiffs = column.map(value => {
    const diff = value - avg;
    const sqrDiff = diff * diff;
    return sqrDiff;
  });

  let avgSquareDiff = calculateMean(squareDiffs);

  const stdDev = Math.sqrt(avgSquareDiff);*/
  return std(column);
};

function normalize(value, columnMean, columnStdDev) {
  return divide(subtract(value, columnMean), columnStdDev);
}

export { calculateMean, calculateStdDev, normalize };
