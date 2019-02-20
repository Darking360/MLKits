const tf = require('@tensorflow/tfjs')

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
  }

  train() {
    for(let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }
}

module.exports = LinearRegression