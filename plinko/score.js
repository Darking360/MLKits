const outputs = [];

function distance(pointA, pointB) {
  // point A and B would be arrays
  return _.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => (a-b) ** 2)
    .sum()
    .value() ** 0.5
}

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 10;

    _.range(1, 15).forEach(k => {
      const [testSet, trainingSet] = splitDataset(outputs, testSetSize, k);
      const accuracy = _.chain(testSet)
        .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3] )
        .size()
        .divide(testSetSize)
        .value()

        console.log(`Accuracy for K = '${k}': ${accuracy}`)
    });
}

function knn(data, point, k) {
  // Point has 3 values
  return _.chain(data)
  .map(row => {
    return [
        distance(_.initial(row), point), 
        _.last(row)
      ]
  })
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

