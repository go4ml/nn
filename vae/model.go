/*
Package vae implements Auto-Encoding Variational Bayes Algorithm

https://arxiv.org/pdf/1312.6114.pdf
*/
package vae

import (
	"go-ml.dev/pkg/base/model"
	"go-ml.dev/pkg/base/model/hyperopt"
	"go-ml.dev/pkg/nn"
	"go-ml.dev/pkg/nn/mx"
	"reflect"
)

// Latent is the default name of feature for the decoder
const LatentCol = "Latent"

// default batch size for auto-encoders training
const DefaultBatchSize = 32

/*
Model of the Variational Auto-Encoder
*/
type Model struct {
	// size of hidden layer, half of input by default
	Hidden int
	// size of latent (encoder Output/decoder input) layer
	Latent int
	// latent layer tensor as output of encoder and input for decoder
	// vae.LatentCol by default
	Feature string
	// generative output for decoder
	// model.PredictedCol by default
	Predicted string
	// Mxnet Context
	// mx.CPU by default
	Context mx.Context
	// batch size
	// vae.DefaultBatchSize by default
	BatchSize int
	// random generator seed
	// random by default
	Seed int
	// optimizer config
	// nn.Adam{Lr:0.001} by default
	Optimizer nn.OptimizerConf
	// input width
	// normally it's calculated from features
	Width int
	// B-VAE hyper-patameter
	Beta float64
}

/*
Feed model with dataset
*/
func (e Model) Feed(ds model.Dataset) model.FatModel {
	return func(workout model.Workout) (*model.Report, error) {
		return train(e, ds, workout)
	}
}

/*
EncoderCollection is the name of collection containing encoder model
*/
const EncoderCollection = "encoder"

/*
DecoderCollection is the name of collection containing decoder model
*/
const DecoderCollection = "decoder"

/*
RecoderCollection is the name of collection containing recoder model
*/
const RecoderCollection = "recoder"

/*
ModelFunc updates model with parameters for hyper-optimization
*/
func (e Model) ModelFunc(params hyperopt.Params) model.HungryModel {
	return e.Apply(params)
}

/*
Apply parameters to define model specific
*/
func (e Model) Apply(params hyperopt.Params) Model {
	hyperopt.Apply(params, map[string]reflect.Value{
		"Hidden": reflect.ValueOf(&e.Hidden),
		"Latent": reflect.ValueOf(&e.Latent),
		"Beta":   reflect.ValueOf(&e.Beta),
		"Seed":   reflect.ValueOf(&e.Seed),
	})
	return e
}
