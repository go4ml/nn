package nn

import "go-ml.dev/pkg/nn/mx"

type Dropout struct {
	Rate float32
}

func (ly *Dropout) Combine(in *mx.Symbol) *mx.Symbol {
	out := in
	if ly.Rate > 0.01 {
		out = mx.Dropout(out, ly.Rate)
	}
	return out
}
