package nn

import "go4ml.xyz/nn/mx"

type Lambda struct {
	F func(*mx.Symbol) *mx.Symbol
}

func (nb Lambda) Combine(input *mx.Symbol) *mx.Symbol {
	return nb.F(input)
}
