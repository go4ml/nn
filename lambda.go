package nn

import "go-ml.dev/pkg/nn/mx"

type Lambda struct {
	F func(*mx.Symbol) *mx.Symbol
}

func (nb Lambda) Combine(input *mx.Symbol) *mx.Symbol {
	return nb.F(input)
}
