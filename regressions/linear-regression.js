const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features
    this.labels = labels

    this.m = 0
    this.b = 0

    this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options)
  }

  
  
  gradientDescent() {
    const currentGuessesForMPG = this.features.map(row => (this.m * row[0]) + this.b)
    const bSlope = _.sum(currentGuessesForMPG.map((guess, index) => guess - this.labels[index][0])) * 2 / this.features.length
    const mSlope = _.sum(currentGuessesForMPG.map((guess, index) => -1 * this.features[index][0] * (this.labels[index][0] - guess))) * 2 / this.features.length

    this.m = this.m - mSlope * this.options.learningRate
    this.b = this.b - bSlope * this.options.learningRate
  }


  train() {
    for(let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }
}

module.exports = LinearRegression