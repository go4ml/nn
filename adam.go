package nn

import (
	"go4ml.xyz/nn/mx"
)

type Adam struct {
	Lr, Beta1, Beta2, Epsilon, Decay float64

	LrMap map[int]float64
}

type stAdam struct {
	Var   *mx.NDArray
	Mean  *mx.NDArray
	Index int
}

type implAdam struct {
	Adam
	States map[*mx.NDArray]stAdam
}

func (opt Adam) Init(e int) Optimizer {
	r := &implAdam{Adam: opt, States: make(map[*mx.NDArray]stAdam)}
	if r.Lr == 0 {
		r.Lr = locateLr(e, opt.LrMap, 0.001)
	}
	if r.Beta1 == 0 {
		r.Beta1 = 0.9
	}
	if r.Beta2 == 0 {
		r.Beta2 = 0.999
	}
	if r.Epsilon == 0 {
		r.Epsilon = 1e-8
	}
	return r
}

func (opt *implAdam) Release() {
	for _, v := range opt.States {
		v.Var.Release()
		v.Mean.Release()
	}
}

func (opt *implAdam) Update(params *mx.NDArray, grads *mx.NDArray) {
	st, ok := opt.States[params]
	if !ok {
		v := params.NewLikeThis().Zeros()
		m := params.NewLikeThis().Zeros()
		st = stAdam{Var: v, Mean: m}
		opt.States[params] = st
	}
	mx.AdamUpdate(params, grads, st.Mean, st.Var, opt.Lr, opt.Beta1, opt.Beta2, opt.Epsilon, opt.Decay)
}
