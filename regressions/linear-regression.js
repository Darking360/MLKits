const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
  constructor(features, labels, options) {
    this.features = tf.tensor(features)
    this.labels = tf.tensor(labels)

    this.features = tf.ones([ this.features.shape[0], 1 ]).concat(this.features, 1)

    this.weights = tf.zeros([2,1])

    this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options)
  }

  /*gradientDescent() {
    const currentGuessesForMPG = this.features.map(row => (this.m * row[0]) + this.b)
    const bSlope = _.sum(currentGuessesForMPG.map((guess, index) => guess - this.labels[index][0])) * 2 / this.features.length
    const mSlope = _.sum(currentGuessesForMPG.map((guess, index) => -1 * this.features[index][0] * (this.labels[index][0] - guess))) * 2 / this.features.length

    this.m = this.m - mSlope * this.options.learningRate
    this.b = this.b - bSlope * this.options.learningRate
  }*/

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights)
    const differences = currentGuesses.sub(this.labels)
    const slopes = this.features.transpose()
      .matMul(differences)
      .div(this.features.shape[0])
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
  }


  train() {
    for(let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = tf.tensor(testFeatures)
    testLabels = tf.tensor(testLabels)
    testFeatures = tf.ones([ testFeatures.shape[0], 1 ]).concat(testFeatures, 1)
    const predictions = testFeatures.matMul(this.weights)
    
    const SSRes = testLabels.sub(predictions)
      .pow(2)
      .sum()
      .get()
      
    const SSTot = testLabels.sub(testLabels.mean())
      .pow(2)
      .sum()
      .get()

    return 1 - SSRes / SSTot
  }
}

module.exports = LinearRegression