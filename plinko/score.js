const outputs = [];
const predictionPoint = 300; 
const k = 3;

function distance(point) {
  return Math.abs(point - predictionPoint);
}

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  // Write code here to analyze stuff
  // Implementing
  const bucket = _.chain(outputs)
    .map(row => [distance(row[0]), row[3]])
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .parseInt()
    .value()

  console.log('Your point will probably fall into bucket #' + bucket)
}

