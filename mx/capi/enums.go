package capi

type MxnetKey int

const (
	KeyEmpty MxnetKey = iota
	KeyLow
	KeyHigh
	KeyScalar
	KeyLhs
	KeyRhs
	KeyData
	KeyExclude
	KeyAxis
	KeyAxes
	KeyBegin
	KeyEnd
	KeyMode
	KeyNormalization
	KeyLr
	KeyMomentum
	KeyWd
	KeyBeta1
	KeyBeta2
	KeyEpsilon
	KeyEps
	KeyLoc
	KeyScale
	KeyKeepdims
	KeyNoBias
	KeyNumGroup
	KeyNumFilter
	KeyKernel
	KeyStride
	KeyPad
	KeyActType
	KeyPoolType
	KeyPoolConv
	KeyFlatten
	KeyNumHidden
	KeyMultiOutput
	KeyNumArgs
	KeyLayout
	KeyShape
	KeyGlobalStats
	KeyP
	KeyDim1
	KeyDim2
	KeyNoKey
)

var keymap = map[MxnetKey]string{
	KeyLow:           "low",
	KeyHigh:          "high",
	KeyScalar:        "scalar",
	KeyLhs:           "lhs",
	KeyRhs:           "rhs",
	KeyData:          "data",
	KeyExclude:       "exclude",
	KeyAxis:          "axis",
	KeyAxes:          "axes",
	KeyBegin:         "begin",
	KeyEnd:           "end",
	KeyMode:          "mode",
	KeyLr:            "lr",
	KeyMomentum:      "momentum",
	KeyWd:            "wd",
	KeyBeta1:         "beta1",
	KeyBeta2:         "beta2",
	KeyEpsilon:       "epsilon",
	KeyEps:           "eps",
	KeyNormalization: "normalization",
	KeyLoc:           "loc",
	KeyScale:         "scale",
	KeyKeepdims:      "keepdims",
	KeyNoBias:        "no_bias",
	KeyNumGroup:      "num_group",
	KeyNumFilter:     "num_filter",
	KeyKernel:        "kernel",
	KeyStride:        "stride",
	KeyPad:           "pad",
	KeyActType:       "act_type",
	KeyPoolType:      "pool_type",
	KeyPoolConv:      "pooling_convention",
	KeyFlatten:       "flatten",
	KeyMultiOutput:   "multi_output",
	KeyNumHidden:     "num_hidden",
	KeyNumArgs:       "num_args",
	KeyLayout:        "layout",
	KeyGlobalStats:   "use_global_stats",
	KeyShape:         "shape",
	KeyP:             "p",
	KeyDim1:          "dim1",
	KeyDim2:          "dim2",
}

func (k MxnetKey) Value() string {
	if v, ok := keymap[k]; ok {
		return v
	}
	if k == KeyEmpty {
		return ""
	}
	panic("mxnet parameters key out of range")
}

type MxnetOp int

const (
	OpEmpty MxnetOp = iota
	OpRandomUniform
	OpRandomNormal
	OpCopyTo
	OpAdd
	OpAddScalar
	OpSub
	OpSubScalar
	OpSubScalarR
	OpMul
	OpMulScalar
	OpDiv
	OpDivScalar
	OpDivScalarR
	OpMean
	OpStack
	OpAbs
	OpBlockGrad
	OpMakeLoss
	OpZeros
	OpZerosLike
	OpOnes
	OpOnesLike
	OpPowerScalar
	OpPowerScalarR
	OpSgdUpdate
	OpSgdMomUpdate
	OpAdamUpdate
	OpLogSoftmax
	OpSoftmax
	OpSoftmaxOutput
	OpSoftmaxCE
	OpSoftmaxAC
	OpSum
	OpSumNan
	OpDot
	OpPick
	OpSquare
	OpSqrt
	OpConcat
	OpConvolution
	OpActivation
	OpPooling
	OpFullyConnected
	OpFlatten
	OpLog
	OpCosh
	OpNot
	OpSigmoid
	OpHardSigmoid
	OpTanh
	OpSin
	OpReLU
	OpBatchNorm
	OpBroadcastMul
	OpBroadcastDiv
	OpBroadcastSub
	OpBroadcastAdd
	OpTranspose
	OpSlice
	OpLeScalar
	OpGeScalar
	OpNeScalar
	OpEqScalar
	OpLesserScalar
	OpGreaterScalar
	OpLe
	OpGe
	OpNe
	OpEq
	OpLesser
	OpGreater
	OpReshape
	OpReshapeLike
	OpAnd
	OpOr
	OpXor
	OpDropout
	OpExp
	OpSwapAxis
	OpNoOp
)

var opmap = map[MxnetOp]string{
	OpRandomUniform:  "_random_uniform",
	OpRandomNormal:   "_random_normal",
	OpCopyTo:         "_copyto",
	OpAdd:            "elemwise_add",
	OpAddScalar:      "_plus_scalar",
	OpSub:            "elemwise_sub",
	OpSubScalar:      "_minus_scalar",
	OpSubScalarR:     "_rminus_scalar",
	OpMul:            "elemwise_mul",
	OpMulScalar:      "_mul_scalar",
	OpDiv:            "elemwise_div",
	OpDivScalar:      "_div_scalar",
	OpDivScalarR:     "_rdiv_scalar",
	OpMean:           "mean",
	OpStack:          "stack",
	OpAbs:            "abs",
	OpBlockGrad:      "BlockGrad",
	OpMakeLoss:       "make_loss",
	OpZeros:          "_zeros",
	OpZerosLike:      "zeros_like",
	OpOnes:           "_ones",
	OpOnesLike:       "ones_like",
	OpPowerScalar:    "_power_scalar",
	OpPowerScalarR:   "_rpower_scalar",
	OpSgdUpdate:      "sgd_update",
	OpSgdMomUpdate:   "sgd_mom_update",
	OpAdamUpdate:     "adam_update",
	OpLogSoftmax:     "log_softmax",
	OpSoftmax:        "softmax",
	OpSoftmaxCE:      "softmax_cross_entropy",
	OpSoftmaxAC:      "SoftmaxActivation",
	OpSoftmaxOutput:  "SoftmaxOutput",
	OpSum:            "sum",
	OpSumNan:         "nansum",
	OpDot:            "dot",
	OpPick:           "pick",
	OpSquare:         "square",
	OpSqrt:           "sqrt",
	OpConcat:         "Concat",
	OpConvolution:    "Convolution",
	OpActivation:     "Activation",
	OpPooling:        "Pooling",
	OpFullyConnected: "FullyConnected",
	OpFlatten:        "Flatten",
	OpNot:            "logical_not",
	OpAnd:            "_logical_and",
	OpOr:             "_logical_or",
	OpXor:            "_logical_xor",
	OpLog:            "log",
	OpCosh:           "cosh",
	OpSin:            "sin",
	OpTanh:           "tanh",
	OpSigmoid:        "sigmoid",
	OpHardSigmoid:    "hard_sigmoid",
	OpReLU:           "relu",
	OpBroadcastSub:   "broadcast_sub",
	OpBroadcastAdd:   "broadcast_add",
	OpBroadcastMul:   "broadcast_mul",
	OpBroadcastDiv:   "broadcast_div",
	OpTranspose:      "transpose",
	OpSlice:          "slice",
	OpLe:             "_lesser_equal",
	OpGe:             "_greater_equal",
	OpNe:             "_not_equal",
	OpEq:             "_equal",
	OpLesser:         "_lesser",
	OpGreater:        "_greater",
	OpLeScalar:       "_lesser_equal_scalar",
	OpGeScalar:      "_greater_equal_scalar",
	OpNeScalar:      "_not_equal_scalar",
	OpEqScalar:      "_equal_scalar",
	OpLesserScalar:  "_lesser_scalar",
	OpGreaterScalar: "_greater_scalar",
	OpReshape:       "Reshape",
	OpReshapeLike:   "reshape_like",
	OpBatchNorm:     "BatchNorm",
	OpDropout:       "Dropout",
	OpExp:           "exp",
	OpSwapAxis:      "SwapAxis",
}

func (o MxnetOp) Value() string {
	if v, ok := opmap[o]; ok {
		return v
	}
	panic("mxnet operation out of range")
}
