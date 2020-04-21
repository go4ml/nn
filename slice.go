package nn

import (
	"fmt"
	"go-ml.dev/pkg/nn/mx"
)

type Slice struct {
	Axis    int
	Begin   int
	End     int
	Name    string
	Output  bool
	TurnOff bool
}

func (ly Slice) Combine(in *mx.Symbol) *mx.Symbol {
	if ly.TurnOff {
		return in
	}

	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("Slice%02d", NextSymbolId())
	}
	out := mx.Slice(in, ly.Axis, ly.Begin, ly.End)
	out.SetName(ns)
	out.SetOutput(ly.Output)
	return out
}
