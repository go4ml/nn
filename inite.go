package nn

import "go4ml.xyz/nn/mx"

type Const struct {
	Value float32
}

func (x Const) Inite(a *mx.NDArray) {
	if x.Value == 0 {
		a.Zeros()
	}
	a.Fill(x.Value)
}

type XavierFactor int

const (
	XavierIn  XavierFactor = 1
	XavierOut XavierFactor = 2
	XavierAvg XavierFactor = 3
)

type Xavier struct {
	Gaussian  bool
	Magnitude float32
	Factor    XavierFactor
}

func (x Xavier) Inite(a *mx.NDArray) {
	var magnitude float32 = 3.
	if x.Magnitude > 0 {
		magnitude = x.Magnitude
	}
	factor := 2 // Avg
	if x.Factor >= XavierIn && x.Factor <= XavierAvg {
		factor = int(x.Factor)
	}
	a.Xavier(x.Gaussian, factor, magnitude)
}

type Uniform struct {
	Magnitude float32
}

func (x Uniform) Inite(a *mx.NDArray) {
	var magnitude float32 = 1.
	if x.Magnitude > 0 {
		magnitude = x.Magnitude
	}
	a.Uniform(0, magnitude)
}
