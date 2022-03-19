package nn

import "go4ml.xyz/nn/mx"

type SGD struct {
	Lr, Mom, Decay float64

	LrMap map[int]float64
}

func (opt SGD) Init(e int) Optimizer {
	r := &implSGD{SGD: opt, States: make(map[*mx.NDArray]*mx.NDArray)}
	if r.Lr == 0 {
		r.Lr = locateLr(e, opt.LrMap, 0.01)
	}
	return r
}

type implSGD struct {
	SGD
	States map[*mx.NDArray]*mx.NDArray
}

func (opt *implSGD) Release() {
	for _, v := range opt.States {
		v.Release()
	}
}

func (opt *implSGD) Update(params *mx.NDArray, grads *mx.NDArray) {
	if opt.Mom != 0 {
		st, ok := opt.States[params]
		if !ok {
			st = params.NewLikeThis().Zeros()
			opt.States[params] = st
		}
		mx.SgdMomUpdate(params, grads, st, opt.Lr, opt.Mom, 0)
	}
	mx.SgdUpdate(params, grads, opt.Lr, opt.Decay)
}
