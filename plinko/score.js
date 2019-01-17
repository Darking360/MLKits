const outputs = [];

function distance(point, predictionPoint) {
  return Math.abs(point - predictionPoint);
}

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 100;

    _.range(1, 15).forEach(k => {
      const [testSet, trainingSet] = splitDataset(outputs, testSetSize, k);
      const accuracy = _.chain(testSet)
        .filter(testPoint => knn(trainingSet, testPoint[0]) === testPoint[3] )
        .size()
        .divide(testSetSize)
        .value()

        console.log(`Accuracy for K = '${k}': ${accuracy}`)
    });
}

function knn(data, point, k) {
  return _.chain(data)
  .map(row => [distance(row[0], point), row[3]])
  .sortBy(row => row[0])
  .slice(0, k)
  .countBy(row => row[1])
  .toPairs()
  .sortBy(row => row[1])
  .last()
  .parseInt()
  .value()
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0 , testCount);
  const trainingSet = _.slice(shuffled, testCount);
  return [testSet, trainingSet];
}

