package nn

import (
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/nn/mx"
)

type L0Loss struct{}

func (L0Loss) Loss(out *mx.Symbol) *mx.Symbol {
	return out
}

type L1Loss struct{ Num int }

func (loss L1Loss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.Ifei(loss.Num == 0, 1, loss.Num)
	label := mx.Var("_label", mx.Dim(0, n))
	return mx.Mean(mx.Abs(mx.Sub(out, label)))
}

type L2Loss struct{ Num int }

func (loss L2Loss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.Ifei(loss.Num == 0, 1, loss.Num)
	label := mx.Var("_label", mx.Dim(0, n))
	return mx.Square(mx.Sub(out, label))
}

type SoftmaxCrossEntropyLoss struct{}

func (SoftmaxCrossEntropyLoss) Loss(out *mx.Symbol) *mx.Symbol {
	label := mx.Var("_label", mx.Dim(0, 1))
	return mx.SoftmaxCrossEntropy(out, label)
}

type CrossEntropyLoss struct{ Num int }

func (loss CrossEntropyLoss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.Ifei(loss.Num == 0, 1, loss.Num)
	label := mx.Var("_label", mx.Dim(0, n))
	a := mx.Log(mx.Add(mx.Pick(out, label), 1e-12))
	return mx.Sum(mx.Mul(a, -1), -1)
}

type LcosLoss struct{ Num int }

func (loss LcosLoss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.Ifei(loss.Num == 0, 1, loss.Num)
	label := mx.Var("_label", mx.Dim(0, n))
	return mx.LogCosh(mx.Sub(out, label))
}

type LossFunc func(*mx.Symbol) *mx.Symbol

func (loss LossFunc) Loss(out *mx.Symbol) *mx.Symbol {
	return loss(out)
}
