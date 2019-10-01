"use strict";

const calculateMean = column => {
  const sum = column.reduce((a, b) => a + b);
  const avg = sum / column.length;
  return avg;
};

const calculateStdDev = column => {
  const avg = calculateMean(column);

  const squareDiffs = column.map(value => {
    const diff = value - avg;
    const sqrDiff = diff * diff;
    return sqrDiff;
  });

  let avgSquareDiff = calculateMean(squareDiffs);

  const stdDev = Math.sqrt(avgSquareDiff);
  return stdDev;
};

export default function normalize(value, column) {
  const mean = calculateMean(column);
  const stdDev = calculateStdDev(column);

  const normalizedValue = (value - mean) / stdDev;
  return { normalizedValue, mean, stdDev };
}
