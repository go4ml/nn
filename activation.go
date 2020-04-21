package nn

import (
	"fmt"
	"go-ml.dev/pkg/nn/mx"
)

func Sigmoid(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.ActivSigmoid)
}

func HardSigmoid(a *mx.Symbol) *mx.Symbol {
	return mx.HardSigmoid(a)
}

func Tanh(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.ActivTanh)
}

func Tanh25(a *mx.Symbol) *mx.Symbol {
	return mx.Add(mx.Mul(mx.Activation(a, mx.ActivTanh), 0.5), 0.5)
}

func ReLU(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.ActivReLU)
}

func SoftReLU(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.ActivSoftReLU)
}

func SoftSign(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.ActivSoftSign)
}

func Softmax(a *mx.Symbol) *mx.Symbol {
	return mx.SoftmaxActivation(a, false)
}

func ChannelSoftmax(a *mx.Symbol) *mx.Symbol {
	return mx.SoftmaxActivation(a, true)
}

func Swish(a *mx.Symbol) *mx.Symbol {
	return mx.Mul(mx.Sigmoid(a), a)
}

func Sin(a *mx.Symbol) *mx.Symbol {
	return mx.Sin(a)
}

type Activation struct {
	Function  func(*mx.Symbol) *mx.Symbol
	BatchNorm bool
	Name      string
}

func (ly Activation) Combine(in *mx.Symbol) *mx.Symbol {
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("Activation%02d", NextSymbolId())
	} else {
		ns += "$A"
	}
	out := in
	if ly.BatchNorm {
		out = BatchNorm{Name: ly.Name}.Combine(in)
	}
	if ly.Function != nil {
		out = ly.Function(out)
	}
	out.SetName(ns)
	return out
}
