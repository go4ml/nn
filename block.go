package nn

import (
	"go-ml.dev/pkg/nn/mx"
)

type Block interface {
	Combine(*mx.Symbol) *mx.Symbol
}

func Combine(nn Block) *mx.Symbol {
	symbolMu.Lock()
	defer symbolMu.Unlock()
	resetSymbolId(0)
	return nn.Combine(mx.Input())
}

type BlockConnect struct {
	blocks []Block
}

func (bc *BlockConnect) Combine(s *mx.Symbol) *mx.Symbol {
	for _, b := range bc.blocks {
		s = b.Combine(s)
	}
	return s
}

func Sequence(b ...Block) Block {
	return &BlockConnect{b}
}

type BlockConcat struct {
	blocks []Block
}

func (bc *BlockConcat) Combine(s *mx.Symbol) *mx.Symbol {
	b := make([]*mx.Symbol, 0, len(bc.blocks))
	for _, v := range bc.blocks {
		if v != nil {
			x := v.Combine(s)
			b = append(b, x)
		}
	}
	return mx.Concat(b...)
}

func Concat(b ...Block) Block {
	return &BlockConcat{b}
}

type BlockStack struct {
	blocks []Block
	axis1  bool
}

func (bc *BlockStack) Combine(s *mx.Symbol) *mx.Symbol {
	b := make([]*mx.Symbol, len(bc.blocks), len(bc.blocks))
	for i, v := range bc.blocks {
		b[i] = v.Combine(s)
	}
	if bc.axis1 {
		return mx.Stack1(b...)
	}
	return mx.Stack(b...)
}

func TransStack(b ...Block) Block {
	return &BlockStack{b, true}
}

func Stack(b ...Block) Block {
	return &BlockStack{b, false}
}

type ResidualBlock struct {
	blocks []Block
}

func Residual(a ...Block) Block {
	return &ResidualBlock{a}
}

func (rcb *ResidualBlock) Combine(a *mx.Symbol) *mx.Symbol {
	for _, n := range rcb.blocks {
		a = mx.Add(a, n.Combine(a))
	}
	return a
}
