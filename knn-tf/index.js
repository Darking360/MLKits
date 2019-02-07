require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
	shuffle: true,
	splitTest: 10,
	dataColumns: ['lat', 'long'],
	labelColumns: ['price']
});

function knn(features, labels, predictionPoint, k) {
	return features
		.sub(predictionPoint)
		.pow(2)
		.sum(1)
		.pow(.5)
		.expandDims(1)
		.concat(labels, 1)
		.unstack()
		.sort((a,b) => {
			return a.get(0) > b.get(0) ? 1 : -1
		})
		.slice(0, k)
		.reduce((acc, pair) => acc += pair.get(1) , 0) / k
}

function getErrorRate(expected, actual) {
	return ((expected - actual) / expected) * 100
}

console.log(testFeatures);
console.log(testLabels);

features = tf.tensor(features)
labels = tf.tensor(labels)

testFeatures.forEach((testPoint, i) => {
	const result = knn(features, labels, tf.tensor(testPoint), 10)
	console.log('Guess', result, 'supposed to be:', testLabels[i][0], 'error rate far %:', getErrorRate(testLabels[i][0], result))
})


