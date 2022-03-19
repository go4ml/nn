package nn

import (
	"fmt"
	"go4ml.xyz/nn/mx"
)

type Convolution struct {
	Channels   int
	Kernel     mx.Dimension
	Stride     mx.Dimension
	Padding    mx.Dimension
	Activation func(*mx.Symbol) *mx.Symbol
	WeightInit mx.Inite // none by default
	BiasInit   mx.Inite // &nn.Const{0} by default
	NoBias     bool
	Groups     bool
	BatchNorm  bool
	Layout     string
	Name       string
	Round      int
	TurnOff    bool
	Output     bool
	Dropout    float32
}

func (ly Convolution) Combine(in *mx.Symbol) *mx.Symbol {
	var bias *mx.Symbol

	if ly.TurnOff {
		return in
	}

	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("Conv%02d", NextSymbolId())
	}
	weight := mx.Var(ns+"_weight", ly.WeightInit)
	if !ly.NoBias {
		init := ly.BiasInit
		if init == nil {
			init = Uniform{0.01}
		}
		bias = mx.Var(ns+"_bias", init)
	}
	k := ly.Kernel
	if k.Len == 0 {
		k = mx.Dim(1, 1)
	}
	out := mx.Conv(in, weight, bias, ly.Channels, k, ly.Stride, ly.Padding, ly.Groups, ly.Layout)
	if ly.Round != 0 {
		ns += fmt.Sprintf("$RNN%02d", ly.Round)
	}
	out.SetName(ns)
	if ly.BatchNorm && ly.Round == 0 {
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

type MaxPool struct {
	Kernel  mx.Dimension
	Stride  mx.Dimension
	Padding mx.Dimension
	Ceil    bool
	Name    string
	Round   int

	BatchNorm bool
}

func (ly MaxPool) Combine(in *mx.Symbol) *mx.Symbol {
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("MaxPool%02d", NextSymbolId())
	}
	out := mx.Pool(in, ly.Kernel, ly.Stride, ly.Padding, ly.Ceil, true)
	if ly.Round != 0 {
		ns += fmt.Sprintf("$RNN%02d", ly.Round)
	}
	out.SetName(ns)
	if ly.BatchNorm && ly.Round == 0 {
		out = BatchNorm{Name: ns}.Combine(out)
	}
	return out
}

type AvgPool struct {
	Kernel  mx.Dimension
	Stride  mx.Dimension
	Padding mx.Dimension
	Ceil    bool
	Name    string
	Round   int

	BatchNorm bool
}

func (ly AvgPool) Combine(in *mx.Symbol) *mx.Symbol {
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("AvgPool%02d", NextSymbolId())
	}
	out := mx.Pool(in, ly.Kernel, ly.Stride, ly.Padding, ly.Ceil, false)
	if ly.Round != 0 {
		ns += fmt.Sprintf("$RNN%02d", ly.Round)
	}
	out.SetName(ns)
	if ly.BatchNorm && ly.Round == 0 {
		out = BatchNorm{Name: ns}.Combine(out)
	}
	return out
}
