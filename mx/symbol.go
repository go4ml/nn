package mx

import (
	"fmt"
	"go4ml.xyz/nn/mx/capi"
	"strings"
)

const (
	OpVar_    capi.MxnetOp = -1
	OpInput_  capi.MxnetOp = -2
	OpScalar_ capi.MxnetOp = -4
	OpNogVar_ capi.MxnetOp = -5
	OpGroup_  capi.MxnetOp = -7
	OpRef_    capi.MxnetOp = -8
	OpOutput_ capi.MxnetOp = -9
	OpBound_  capi.MxnetOp = -10
	OpDepend_ capi.MxnetOp = -11
	OpLink_   capi.MxnetOp = -12
)

type Inite interface {
	Inite(*NDArray)
}

type _Value struct{ Value []float32 }

func (v *_Value) Inite(arr *NDArray) {
	arr.SetValues(v.Value)
}

type Symbol struct {
	Op     capi.MxnetOp             `yaml:"op"`
	Value  string                   `yaml:"value"`
	Name   string                   `yaml:"name"`
	Args   []*Symbol                `yaml:"args"`
	Init   Inite                    `yaml:"-"`
	Attr   map[capi.MxnetKey]string `yaml:"attr"`
	Dim    Dimension                `yaml:"dim"`
	Output bool                     `yaml:"output"`
}

type _hidden_input_ struct{}

func Input(..._hidden_input_) *Symbol { return &Symbol{Op: OpInput_} }

type _hidden_nograd_ struct{}

func Nograd(_hidden_nograd_) {}

func (s *Symbol) SetName(name string) *Symbol {
	s.Name = name
	return s
}

func (s *Symbol) SetOutput(on bool) *Symbol {
	s.Output = on
	return s
}

func Output(a *Symbol, name string) *Symbol {
	return &Symbol{Op: OpOutput_, Args: []*Symbol{a}, Name: name}
}

func Bound(a ...*Symbol) *Symbol {
	return &Symbol{Op: OpBound_, Args: a}
}

func Depend(a ...*Symbol) *Symbol {
	return &Symbol{Op: OpDepend_, Args: a}
}

func SymbolCast(i interface{}) (*Symbol, error) {
	var o *Symbol
	switch v := i.(type) {
	case func(..._hidden_input_):
		o = &Symbol{Op: OpInput_}
	case string:
		o = Var(v)
	case *Symbol:
		o = v
	case float32, float64, int, int8, int32, int64, uint, uint8, uint32, uint64:
		o = &Symbol{Op: OpScalar_, Value: fmt.Sprintf("%v", v)}
	}
	if o != nil {
		return o, nil
	}
	return nil, fmt.Errorf("cant cast '%#v' to *Symbol", i)
}

func GenericOp2(op, opScalar, opScalarR capi.MxnetOp, lv interface{}, rv interface{}) *Symbol {
	var (
		l, r *Symbol
		err  error
	)
	if l, err = SymbolCast(lv); err != nil {
		panic(err.Error())
	}
	if r, err = SymbolCast(rv); err != nil {
		panic(err.Error())
	}

	if l != nil && l.Op == OpScalar_ {
		return &Symbol{
			Op:   opScalarR,
			Args: []*Symbol{r},
			Attr: map[capi.MxnetKey]string{capi.KeyScalar: l.Value}}
	}

	if r != nil && r.Op == OpScalar_ {
		return &Symbol{
			Op:   opScalar,
			Args: []*Symbol{l},
			Attr: map[capi.MxnetKey]string{capi.KeyScalar: r.Value}}
	}

	return &Symbol{Op: op, Args: []*Symbol{l, r}}
}

func GenericOp1(op, opScalar capi.MxnetOp, l *Symbol, rv interface{}) *Symbol {
	var (
		r   *Symbol
		err error
	)
	if r, err = SymbolCast(rv); err != nil {
		panic(err.Error())
	}

	if r != nil && r.Op == OpScalar_ {
		return &Symbol{
			Op:   opScalar,
			Args: []*Symbol{l},
			Attr: map[capi.MxnetKey]string{capi.KeyScalar: r.Value}}
	}

	return &Symbol{Op: op, Args: []*Symbol{l, r}}
}

func Add(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpAdd, capi.OpAddScalar, capi.OpAddScalar, lv, rv)
}

func Sub(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpSub, capi.OpSubScalar, capi.OpSubScalarR, lv, rv)
}

func Mul(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpMul, capi.OpMulScalar, capi.OpMulScalar, lv, rv)
}

func Div(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpDiv, capi.OpDivScalar, capi.OpDivScalarR, lv, rv)
}

func Dot(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpDot, capi.OpEmpty, capi.OpEmpty, lv, rv)
}

func LE(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpLe, capi.OpLeScalar, a, rv)
}

func GE(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpGe, capi.OpGeScalar, a, rv)
}

func EQ(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpEq, capi.OpEqScalar, a, rv)
}

func NE(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpNe, capi.OpNeScalar, a, rv)
}

func Lesser(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpLesser, capi.OpLesserScalar, a, rv)
}

func Greater(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpGreater, capi.OpGreaterScalar, a, rv)
}

func And(a *Symbol, b *Symbol) *Symbol {
	return &Symbol{Op: capi.OpAnd, Args: []*Symbol{a, b}}
}

func Or(a *Symbol, b *Symbol) *Symbol {
	return &Symbol{Op: capi.OpOr, Args: []*Symbol{a, b}}
}

func Xor(a *Symbol, b *Symbol) *Symbol {
	return &Symbol{Op: capi.OpXor, Args: []*Symbol{a, b}}
}

func BcastAdd(a, b *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpBroadcastAdd,
		Args: []*Symbol{a, b},
	}
}

func BcastMul(a, b *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpBroadcastMul,
		Args: []*Symbol{a, b},
	}
}

func BcastDiv(a, b *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpBroadcastDiv,
		Args: []*Symbol{a, b},
	}
}

func BcastSub(a, b *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpBroadcastSub,
		Args: []*Symbol{a, b},
	}
}

func Log(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpLog, Args: []*Symbol{a}}
}

func Cosh(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpCosh, Args: []*Symbol{a}}
}

func LogCosh(a *Symbol) *Symbol {
	return Log(Cosh(a))
}

func Not(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpNot, Args: []*Symbol{a}}
}

func Var(name string, opt ...interface{}) *Symbol {
	s := &Symbol{Op: OpVar_, Name: name}
	for _, t := range opt {
		if t == nil {
			continue
		} else if init, ok := t.(Inite); ok {
			s.Init = init
		} else if _, ok := t.(func(_hidden_nograd_)); ok {
			s.Op = OpNogVar_
		} else if dim, ok := t.(Dimension); ok {
			s.Dim = dim
		} else {
			panic(fmt.Sprintf("unexpected parameter %v", t))
		}
	}
	return s
}

func Value(name string, a ...float32) *Symbol {
	return Var(name, Dim(len(a)), &_Value{Value: a})
}

func Link(name string) *Symbol {
	return &Symbol{Op: OpLink_, Name: name}
}

func Ref(name string, a ...*Symbol) *Symbol {
	return &Symbol{Op: OpRef_, Name: name, Args: a}
}

func Group(a ...*Symbol) *Symbol {
	return &Symbol{Op: OpGroup_, Args: a}
}

func MakeLoss(s *Symbol) *Symbol {
	return &Symbol{Op: capi.OpMakeLoss, Args: []*Symbol{s}}
}

func BlockGrad(s *Symbol) *Symbol {
	return &Symbol{Op: capi.OpBlockGrad, Args: []*Symbol{s}}
}

func Pow(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpEmpty, capi.OpPowerScalar, capi.OpPowerScalarR, lv, rv)
}

func Abs(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpAbs, Args: []*Symbol{a}}
}

func Square(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpSquare, Args: []*Symbol{a}}
}

func Sqrt(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpSqrt, Args: []*Symbol{a}}
}

func Minus(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpMulScalar, Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{capi.KeyScalar: "-1"}}
}

func Pick(a *Symbol, label *Symbol) *Symbol {
	return &Symbol{Op: capi.OpPick, Args: []*Symbol{a, label},
		Attr: map[capi.MxnetKey]string{capi.KeyKeepdims: "1"}}
}

func LogSoftmax(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpLogSoftmax, Args: []*Symbol{a}}
	if len(axis) >= 1 {
		s.Attr = map[capi.MxnetKey]string{capi.KeyAxis: fmt.Sprintf("%v", axis[0])}
	}
	return s
}

func SoftmaxOutput(a *Symbol, l *Symbol, multiOutput bool) *Symbol {
	s := &Symbol{Op: capi.OpSoftmaxOutput, Args: []*Symbol{a, l}, Attr: map[capi.MxnetKey]string{}}
	if multiOutput {
		s.Attr[capi.KeyMultiOutput] = "1"
	}
	return s
}

func Softmax(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpSoftmax, Args: []*Symbol{a}}
	if len(axis) >= 1 {
		s.Attr = map[capi.MxnetKey]string{capi.KeyAxis: fmt.Sprintf("%v", axis[0])}
	}
	return s
}

func SoftmaxActivation(a *Symbol, channel bool) *Symbol {
	s := &Symbol{Op: capi.OpSoftmaxAC, Args: []*Symbol{a}}
	if channel {
		s.Attr = map[capi.MxnetKey]string{capi.KeyMode: "channel"}
	}
	return s
}

func SoftmaxCrossEntropy(a, b *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpSoftmaxCE, Args: []*Symbol{a, b}}
	if len(axis) >= 1 {
		s.Attr = map[capi.MxnetKey]string{capi.KeyAxis: fmt.Sprintf("%v", axis[0])}
	}
	return s
}

func formatAxis(axis ...int) string {
	if len(axis) == 1 {
		switch axis[0] {
		case 0:
			return "0"
		case 1:
			return "1"
		case -1:
			return "-1"
		default:
			return fmt.Sprintf("%d", axis[0])
		}
	} else {
		s := make([]string, len(axis))
		for i, a := range axis {
			switch a {
			case 0:
				s[i] = "0"
			case 1:
				s[i] = "1"
			case -1:
				s[i] = "-1"
			default:
				s[i] = fmt.Sprintf("%d", a)
			}
		}
		return "(" + strings.Join(s, ",") + ")"
	}
}

func SumNan(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpSumNan, Args: []*Symbol{a}}
	if len(axis) > 0 {
		s.Attr = map[capi.MxnetKey]string{
			capi.KeyAxis: formatAxis(axis...),
		}
	}
	return s
}

func Sum(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpSum, Args: []*Symbol{a}}
	if len(axis) > 0 {
		s.Attr = map[capi.MxnetKey]string{
			capi.KeyAxis: formatAxis(axis...),
		}
	}
	return s
}

func Sum1(a *Symbol) *Symbol {
	s := &Symbol{Op: capi.OpSum, Args: []*Symbol{a}}
	s.Attr = map[capi.MxnetKey]string{
		capi.KeyAxis:     "-1",
		capi.KeyKeepdims: "1",
	}
	return s
}

func SumXl(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpSum, Args: []*Symbol{a}}
	if len(axis) > 0 {
		s.Attr = map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    formatAxis(axis...),
		}
	}
	return s
}

func Mean(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpMean, Args: []*Symbol{a}}
	if len(axis) > 0 {
		s.Attr = map[capi.MxnetKey]string{
			capi.KeyAxis: formatAxis(axis...),
		}
	}
	return s
}

func MeanKd(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpMean, Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{
			capi.KeyKeepdims: "1",
		}}
	if len(axis) > 0 {
		s.Attr[capi.KeyAxis] = formatAxis(axis...)
	}
	return s
}

func MeanXl(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpMean, Args: []*Symbol{a}}
	if len(axis) > 0 {
		s.Attr = map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    formatAxis(axis...),
		}
	}
	return s
}

func Stack(a ...*Symbol) *Symbol {
	s := &Symbol{Op: capi.OpStack, Args: a,
		Attr: map[capi.MxnetKey]string{
			capi.KeyNumArgs: fmt.Sprintf("%d", len(a)),
		}}
	return s
}

func Stack1(a ...*Symbol) *Symbol {
	s := &Symbol{Op: capi.OpStack, Args: a,
		Attr: map[capi.MxnetKey]string{
			capi.KeyNumArgs: fmt.Sprintf("%d", len(a)),
			capi.KeyAxis:    "-1",
		}}
	return s
}

func BatchNorm(a, gamma, beta, rmean, rvar *Symbol, mom, eps float32, useGlobalStats bool, axis ...int) *Symbol {
	s := &Symbol{Op: capi.OpBatchNorm, Args: []*Symbol{a, gamma, beta, rmean, rvar}}
	s.Attr = map[capi.MxnetKey]string{}
	if len(axis) > 0 {
		s.Attr[capi.KeyAxis] = formatAxis(axis...)
	}
	if mom != 0 {
		s.Attr[capi.KeyMomentum] = fmt.Sprintf("%v", mom)
	}
	if eps != 0 {
		s.Attr[capi.KeyEps] = fmt.Sprintf("%v", eps)
	}
	if useGlobalStats {
		s.Attr[capi.KeyGlobalStats] = "1"
	}
	return s
}

func Concat(a ...*Symbol) *Symbol {
	return &Symbol{Op: capi.OpConcat, Args: a,
		Attr: map[capi.MxnetKey]string{capi.KeyNumArgs: fmt.Sprintf("%d", len(a))}}
}

func Conv(a, weight, bias *Symbol, channels int, kernel, stride, padding Dimension, groups bool, layout string) *Symbol {
	args := []*Symbol{a, weight, bias}
	attr := map[capi.MxnetKey]string{capi.KeyNumFilter: fmt.Sprintf("%v", channels)}
	if bias == nil {
		attr[capi.KeyNoBias] = "1"
	}
	if groups {
		attr[capi.KeyNumGroup] = "2"
	}

	if kernel.Len > 1 {
		attr[capi.KeyKernel] = kernel.String()
	} else if kernel.Len == 1 {
		attr[capi.KeyKernel] = fmt.Sprintf("%v", kernel.Shape[0])
	}

	if stride.Len > 1 {
		attr[capi.KeyStride] = stride.String()
	} else if stride.Len == 1 {
		attr[capi.KeyStride] = fmt.Sprintf("%v", stride.Shape[0])
	}

	if padding.Len > 1 {
		attr[capi.KeyPad] = padding.String()
	} else if padding.Len == 1 {
		attr[capi.KeyPad] = fmt.Sprintf("%v", padding.Shape[0])
	}

	if layout != "" {
		attr[capi.KeyLayout] = layout
	}

	return &Symbol{Op: capi.OpConvolution, Args: args, Attr: attr}
}

type ActivationType int

const (
	ActivReLU ActivationType = iota
	ActivSoftReLU
	ActivSoftSign
	ActivSigmoid
	ActivTanh
)

func Activation(a *Symbol, actType ActivationType) *Symbol {
	var s string
	switch actType {
	case ActivSoftReLU:
		s = "softrelu"
	case ActivSoftSign:
		s = "softsign"
	case ActivSigmoid:
		s = "sigmoid"
	case ActivTanh:
		s = "tanh"
	//case ReLU: s = "relu"
	default:
		s = "relu"
	}
	return &Symbol{Op: capi.OpActivation, Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{capi.KeyActType: s}}
}

func Pool(a *Symbol, kernel, stride, padding Dimension, ceil bool, maxpool bool) *Symbol {
	attr := map[capi.MxnetKey]string{}

	if kernel.Len > 1 {
		attr[capi.KeyKernel] = kernel.String()
	} else if kernel.Len == 1 {
		attr[capi.KeyKernel] = fmt.Sprintf("%v", kernel.Shape[0])
	}

	if stride.Len > 1 {
		attr[capi.KeyStride] = stride.String()
	} else if stride.Len == 1 {
		attr[capi.KeyStride] = fmt.Sprintf("%v", stride.Shape[0])
	}

	if padding.Len > 1 {
		attr[capi.KeyPad] = padding.String()
	} else if padding.Len == 1 {
		attr[capi.KeyPad] = fmt.Sprintf("%v", padding.Shape[0])
	}

	if maxpool {
		attr[capi.KeyPoolType] = "max"
	} else {
		attr[capi.KeyPoolType] = "avg"
	}

	if ceil {
		attr[capi.KeyPoolConv] = "full"
	} else {
		attr[capi.KeyPoolConv] = "valid"
	}

	return &Symbol{Op: capi.OpPooling, Args: []*Symbol{a}, Attr: attr}
}

func FullyConnected(a, weight, bias *Symbol, size int, flatten bool) *Symbol {
	args := []*Symbol{a, weight, bias}
	attr := map[capi.MxnetKey]string{}
	if bias == nil {
		attr[capi.KeyNoBias] = "1"
	}
	if flatten {
		attr[capi.KeyFlatten] = "1"
	}
	attr[capi.KeyNumHidden] = fmt.Sprintf("%v", size)
	return &Symbol{Op: capi.OpFullyConnected, Args: args, Attr: attr}
}

func Flatten(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpFlatten, Args: []*Symbol{a}}
}

func Sigmoid(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpSigmoid, Args: []*Symbol{a}}
}

func HardSigmoid(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpHardSigmoid, Args: []*Symbol{a}}
}

func Tanh(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpTanh, Args: []*Symbol{a}}
}

func Sin(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpSin, Args: []*Symbol{a}}
}

func ReLU(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpReLU, Args: []*Symbol{a}}
}

func Exp(a *Symbol) *Symbol {
	return &Symbol{Op: capi.OpExp, Args: []*Symbol{a}}
}

func Transpose(a *Symbol, axis ...int) *Symbol {
	s := make([]string, len(axis))
	for i, a := range axis {
		switch a {
		case 0:
			s[i] = "0"
		case 1:
			s[i] = "1"
		case -1:
			s[i] = "-1"
		default:
			s[i] = fmt.Sprintf("%d", a)
		}
	}
	ax := "(" + strings.Join(s, ",") + ")"
	return &Symbol{
		Op:   capi.OpTranspose,
		Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{capi.KeyAxes: ax}}
}

func Slice(a *Symbol, axis, begin, end int) *Symbol {
	s := "("
	for i := 0; i < axis; i++ {
		s += "None,"
	}
	return &Symbol{
		Op:   capi.OpSlice,
		Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{
			capi.KeyBegin: fmt.Sprintf(s+"%d)", begin),
			capi.KeyEnd:   fmt.Sprintf(s+"%d)", end),
		}}
}

func Channel(a *Symbol, ch int) *Symbol {
	return &Symbol{
		Op:   capi.OpSlice,
		Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{
			capi.KeyBegin: fmt.Sprintf("(None,%d)", ch),
			capi.KeyEnd:   fmt.Sprintf("(None,%d)", ch+1),
		}}
}

func Ones(dim ...int) *Symbol {
	return &Symbol{
		Op:  capi.OpOnes,
		Dim: Dim(dim...),
	}
}

func Reshape(a *Symbol, dim ...int) *Symbol {
	return &Symbol{
		Op:   capi.OpReshape,
		Dim:  Dim(dim...),
		Args: []*Symbol{a},
	}
}

func OnesLike(a *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpOnesLike,
		Args: []*Symbol{a},
	}
}

func Zeros(dim ...int) *Symbol {
	return &Symbol{
		Op:  capi.OpZeros,
		Dim: Dim(dim...),
	}
}

func ZerosLike(a *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpZerosLike,
		Args: []*Symbol{a},
	}
}

func ReshapeLike(a, b *Symbol) *Symbol {
	return &Symbol{
		Op:   capi.OpReshapeLike,
		Args: []*Symbol{a, b},
	}
}

func SwapAxes(a *Symbol, x, y int) *Symbol {
	return &Symbol{
		Op:   capi.OpSwapAxis,
		Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{
			capi.KeyDim1: fmt.Sprintf("%v", x),
			capi.KeyDim2: fmt.Sprintf("%v", y),
		},
	}
}

func Normal(loc, scale float32, dim ...int) *Symbol {
	return &Symbol{
		Op:  capi.OpRandomNormal,
		Dim: Dim(dim...),
		Attr: map[capi.MxnetKey]string{
			capi.KeyLoc:   fmt.Sprintf("%v", loc),
			capi.KeyScale: fmt.Sprintf("%v", scale),
		},
	}
}

func Dropout(a *Symbol, rate float32) *Symbol {
	return &Symbol{
		Op:   capi.OpDropout,
		Args: []*Symbol{a},
		Attr: map[capi.MxnetKey]string{
			capi.KeyP: fmt.Sprintf("%v", rate),
		},
	}
}
