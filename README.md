[![CircleCI](https://circleci.com/gh/go4ml/nn.svg?style=svg)](https://circleci.com/gh/go4ml/nn)
[![Go Report Card](https://goreportcard.com/badge/github.com/go4ml/nn)](https://goreportcard.com/report/github.com/go4ml/nn)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


```golang

import (
	"fmt"
	"go4ml.xyz/base/model"
	"go4ml.xyz/dataset/mnist"
	"go4ml.xyz/iokit"
	"go4ml.xyz/nn"
	"go4ml.xyz/nn/mx"
	"gotest.tools/assert"
	"testing"
)

var mnistConv0 = nn.Sequence(
	nn.Convolution{Channels: 24, Kernel: mx.Dim(3, 3), Activation: nn.ReLU},
	nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	nn.Convolution{Channels: 32, Kernel: mx.Dim(5, 5), Activation: nn.ReLU, BatchNorm: true},
	nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	nn.FullyConnected{Size: 32, Activation: nn.Swish, BatchNorm: true, Dropout: 0.33},
	nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistConv0(t *testing.T) {
	modelFile := iokit.File(model.Path("mnist_test_conv0.zip"))

	report := nn.Model{
		Network:   mnistConv0,
		Optimizer: nn.Adam{Lr: .001},
		Loss:      nn.CrossEntropyLoss{},
		Input:     mx.Dim(1, 28, 28),
		Seed:      42,
		BatchSize: 32,
		//Context:   mx.GPU,
	}.Feed(model.Dataset{
		Source:   mnist.Data.RandomFlag(model.TestCol, 42, 0.2),
		Label:    model.LabelCol,
		Test:     model.TestCol,
		Features: []string{"Image"},
	}).LuckyTrain(model.Training{
		Iterations: 15,
		ModelFile:  modelFile,
		Metrics:    model.Classification{Accuracy: 0.983},
		Score:      model.ErrorScore,
	})

	fmt.Println(report.TheBest, report.Score)
	fmt.Println(report.History.Round(5))
	assert.Assert(t, model.Accuracy(report.Test) >= 0.98)

	net1 := nn.LuckyObjectify(modelFile) //.Gpu()
	lr := model.LuckyEvaluate(mnist.T10k, model.LabelCol, net1, 32, model.Classification{})
	fmt.Println(lr.Round(5))
	assert.Assert(t, model.Accuracy(lr) >= 0.98)
}
```
