package nn

import (
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/nn/mx"
	"time"
)

type Network struct {
	*mx.Graph
	symbolic  *mx.Symbol
	inputdim  mx.Dimension
	BatchSize int
}

func (network *Network) Release() {
	network.Graph.Release()
}

func New(context mx.Context, nn Block, inputdim mx.Dimension, loss mx.Loss, batchSize int, seed int) *Network {
	symbol := Combine(nn)
	network := &Network{
		Graph:     mx.Compose(context.Upgrade(), symbol, loss, inputdim.Push(batchSize), mx.Float32),
		BatchSize: batchSize,
		symbolic:  symbol,
		inputdim:  inputdim,
	}
	network.Initialize(fu.Seed(seed), nil)
	return network
}

func Load(context mx.Context, symbol, params iokit.Input, batchSize int) (*Network, error) {
	sym, inputdim, err := LoadSymbol(symbol)
	if err != nil {
		return nil, err
	}
	network := &Network{
		Graph:     mx.Compose(context.Upgrade(), sym, nil, inputdim.Push(batchSize), mx.Float32),
		BatchSize: batchSize,
		symbolic:  sym,
		inputdim:  inputdim,
	}
	if err = network.LoadParams(params, true); err != nil {
		return nil, err
	}
	return network, nil
}

func Inherit(context mx.Context, nn Block, inputdim mx.Dimension, params iokit.Input, batchSize int, seed int) (*Network, error) {
	symbol := Combine(nn)
	network := &Network{
		Graph:     mx.Compose(context.Upgrade(), symbol, nil, inputdim.Push(batchSize), mx.Float32),
		BatchSize: batchSize,
		symbolic:  symbol,
		inputdim:  inputdim,
	}
	if seed == 0 {
		seed = int(time.Now().Unix())
	}
	network.Initialize(seed, nil)
	if err := network.LoadParams(params, false); err != nil {
		return nil, err
	}
	return network, nil
}

func (network *Network) Forward(data interface{}, out []float32) {
	network.Graph.Input.SetValues(data)
	network.Graph.Forward(false)
	network.Graph.Output.CopyValuesTo(out)
}

func (network *Network) Predict(data interface{}) [][]float32 {
	out := make([]float32, network.Graph.Output.Dim().Total())
	network.Forward(data, out)
	r := make([][]float32, network.BatchSize)
	stride := len(out) / network.BatchSize
	for i := 0; i < network.BatchSize; i++ {
		r[i] = out[i*stride : (i+1)*stride]
	}
	return r
}

func (network *Network) Train(data interface{}, label interface{}, opt Optimizer) {
	network.Graph.Input.SetValues(data)
	if network.Graph.Label != nil && label != nil {
		network.Graph.Label.SetValues(label)
	}
	network.Graph.Forward(true)
	network.Graph.Backward()
	network.Update(opt)
}

func (network *Network) Update(opt Optimizer) {
	for k, g := range network.Graph.Grads {
		opt.Update(network.Graph.Params[k], g)
	}
}
