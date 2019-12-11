"use strict";

import { mean, std, subtract, divide } from "mathjs";

const calculateMean = column => {
  return mean(column);
};

const calculateStdDev = column => {
  return std(column);
};

function normalize(value, columnMean, columnStdDev) {
  return divide(subtract(value, columnMean), columnStdDev);
}

export { calculateMean, calculateStdDev, normalize };
