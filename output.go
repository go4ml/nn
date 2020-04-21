package nn

import (
	"fmt"
	"go-ml.dev/pkg/nn/mx"
)

type Output struct {
	Name  string
	Round int
	Axis  int
	Begin int
	End   int
}

func (ly Output) Combine(a *mx.Symbol) *mx.Symbol {
	name := ly.Name
	if name == "" {
		name = fmt.Sprintf("Output%d", NextSymbolId())
	}
	if ly.Round > 0 {
		name = fmt.Sprintf("%s$RNN%02d", name, ly.Round)
	}
	out := a
	if ly.Begin != ly.End {
		a = mx.Slice(a, ly.Axis, ly.Begin, ly.End)
		a.SetName(name)
	}
	return mx.Bound(out, mx.Output(a, name))
}
