package nn

import (
	"fmt"
	"go-ml.dev/pkg/nn/mx"
)

type Flatten struct{}

func (ly Flatten) Combine(a *mx.Symbol) *mx.Symbol {
	return mx.Flatten(a)
}

type FullyConnected struct {
	Size       int
	Activation func(*mx.Symbol) *mx.Symbol
	WeightInit mx.Inite // none by default
	BiasInit   mx.Inite // &nn.Const{0} by default
	NoBias     bool
	NoFlatten  bool
	BatchNorm  bool
	Name       string
	Output     bool
	Dropout    float32
}

func (ly FullyConnected) Combine(in *mx.Symbol) *mx.Symbol {
	var bias *mx.Symbol
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("FullyConnected%02d", NextSymbolId())
	}
	weight := mx.Var(ns+"_weight", ly.WeightInit)
	if !ly.NoBias {
		init := ly.BiasInit
		if init == nil {
			init = &Const{0}
		}
		bias = mx.Var(ns+"_bias", init)
	}
	out := mx.FullyConnected(in, weight, bias, ly.Size, !ly.NoFlatten)
	out.SetName(ns)
	if ly.BatchNorm {
		out = BatchNorm{Name: ns}.Combine(out)
	}
	if ly.Activation != nil {
		out = ly.Activation(out)
		out.SetName(ns + "$A")
	}
	if ly.Dropout > 0.01 {
		out = mx.Dropout(out, ly.Dropout)
		out.SetName(ns + "$D")
	}
	out.SetOutput(ly.Output)
	return out
}
