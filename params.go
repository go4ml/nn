package nn

import (
	"bufio"
	"encoding/binary"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/nn/mx"
	"go-ml.dev/pkg/zorros"
	"golang.org/x/xerrors"
	"io"
	"math"
)

func nf(p func(string) bool, f func(string) bool) func(string) bool {
	return func(s string) bool {
		if p(s) {
			return true
		}
		return f(s)
	}
}

func (network *Network) SaveParams(output iokit.Output, only ...string) (err error) {
	patt := func(string) bool { return true }
	if len(only) > 0 {
		patt = func(string) bool { return false }
		for _, o := range only {
			patt = nf(fu.Pattern(o), patt)
		}
	}
	var wr iokit.Whole
	if wr, err = output.Create(); err != nil {
		return zorros.Trace(err)
	}
	defer wr.End()
	params := fu.SortedKeysOf(network.Params).([]string)
	b := []byte{0, 0, 0, 0}
	dil := []byte{0xa, '-', '-', 0xa}
	magic := []byte{'A', 'N', 'N', '1'}
	order := binary.ByteOrder(binary.LittleEndian)
	if _, err = wr.Write(magic); err != nil {
		return zorros.Trace(err)
	}
	count := 0
	for _, n := range params {
		if n[0] != '_' && patt(n) {
			count++
		}
	}
	order.PutUint32(b, uint32(count))
	if _, err = wr.Write(b); err != nil {
		return zorros.Trace(err)
	}
	if _, err = wr.Write(dil); err != nil {
		return zorros.Trace(err)
	}
	for _, n := range params {
		if n[0] == '_' || !patt(n) {
			continue
		}
		d := network.Params[n]
		if err = binary.Write(wr, order, int32(len(n))); err != nil {
			return zorros.Trace(err)
		}
		if err = binary.Write(wr, order, []byte(n)); err != nil {
			return zorros.Trace(err)
		}
		dim := d.Dim()
		order.PutUint32(b, uint32(dim.Len))
		if _, err = wr.Write(b); err != nil {
			return zorros.Trace(err)
		}
		for i := 0; i < dim.Len; i++ {
			order.PutUint32(b, uint32(dim.Shape[i]))
			if _, err = wr.Write(b); err != nil {
				return zorros.Trace(err)
			}
		}
		total := dim.Total()
		order.PutUint32(b, uint32(total))
		if _, err = wr.Write(b); err != nil {
			return zorros.Trace(err)
		}
		v := d.ValuesF32()
		for i := 0; i < total; i++ {
			order.PutUint32(b, math.Float32bits(v[i]))
			if _, err = wr.Write(b); err != nil {
				return zorros.Trace(err)
			}
		}
		if _, err = wr.Write(dil); err != nil {
			return zorros.Trace(err)
		}
	}
	return wr.Commit()
}

var prdDIL = []byte{0xa, '-', '-', 0xa}

type ParamsReader struct {
	io.Closer
	r     io.Reader
	least int
}

func NewParamsReader(input iokit.Input) (prd *ParamsReader, err error) {
	var rd io.ReadCloser
	if rd, err = input.Open(); err != nil {
		return nil, zorros.Trace(err)
	}
	defer func() {
		if err != nil {
			rd.Close()
		}
	}()
	r := bufio.NewReader(rd)
	b := []byte{0, 0, 0, 0}
	equal4b := func(a []byte) bool { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] }
	dil := []byte{0xa, '-', '-', 0xa}
	magic := []byte{'A', 'N', 'N', '1'}
	order := binary.ByteOrder(binary.LittleEndian)
	if _, err = io.ReadFull(r, b); err != nil {
		return nil, zorros.Trace(err)
	}
	if !equal4b(magic) {
		return nil, xerrors.Errorf("bad magic")
	}
	if _, err = io.ReadFull(r, b); err != nil {
		return nil, zorros.Trace(err)
	}
	count := int(order.Uint32(b))
	if _, err = io.ReadFull(r, b); err != nil {
		return nil, zorros.Trace(err)
	}
	if !equal4b(dil) {
		return nil, xerrors.Errorf("bad delimiter")
	}

	prd = &ParamsReader{rd.(io.Closer), r, count}
	return prd, nil
}

func (prd *ParamsReader) HasMore() bool {
	return prd.least > 0
}

func (prd *ParamsReader) Next() (n string, out []float32, err error) {
	b := []byte{0, 0, 0, 0}
	equal4b := func(a []byte) bool { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] }
	order := binary.ByteOrder(binary.LittleEndian)
	var ln int32
	if err = binary.Read(prd.r, order, &ln); err != nil {
		err = zorros.Trace(err)
		return
	}
	ns := make([]byte, ln)
	if err = binary.Read(prd.r, order, &ns); err != nil {
		err = zorros.Trace(err)
		return
	}
	n = string(ns)
	dim := mx.Dimension{}
	if _, err = io.ReadFull(prd.r, b); err != nil {
		err = zorros.Trace(err)
		return
	}
	dim.Len = int(order.Uint32(b))
	if dim.Len > mx.MaxDimensionCount {
		err = xerrors.Errorf("bad deimension of '%v' layer params", n)
		return
	}
	for i := 0; i < dim.Len; i++ {
		if _, err = io.ReadFull(prd.r, b); err != nil {
			err = zorros.Trace(err)
			return
		}
		dim.Shape[i] = int(order.Uint32(b))
	}
	if _, err = io.ReadFull(prd.r, b); err != nil {
		err = zorros.Trace(err)
		return
	}
	total := int(order.Uint32(b))
	if total != dim.Total() {
		err = xerrors.Errorf("bad dimension of '%v' layer params or values total count is incorrect", n)
		return
	}
	v := make([]float32, total)
	for i := range v {
		if _, err = io.ReadFull(prd.r, b); err != nil {
			err = zorros.Trace(err)
			return
		}
		v[i] = math.Float32frombits(order.Uint32(b))
	}
	if _, err = io.ReadFull(prd.r, b); err != nil {
		err = zorros.Trace(err)
		return
	}
	if !equal4b(prdDIL) {
		err = xerrors.Errorf("bad delimiter")
		return
	}
	prd.least--
	return n, v, nil
}

func (network *Network) LoadParams(input iokit.Input, forced ...bool) (err error) {
	var r *ParamsReader
	if r, err = NewParamsReader(input); err != nil {
		return zorros.Trace(err)
	}
	defer r.Close()

	ready := map[string]bool{}

	for r.HasMore() {
		n, v, err := r.Next()
		if err != nil {
			return zorros.Trace(err)
		}
		if d, ok := network.Params[n]; ok {
			if d.Dim().Total() != len(v) {
				return xerrors.Errorf("bad deimension of '%v' layer params or values total count is incorrect", n)
			}
			d.SetValues(v)
			ready[n] = true
		}
	}

	for k := range network.Params {
		if !ready[k] {
			if k[0] != '_' && fu.Fnzb(forced...) {
				return xerrors.Errorf("layer '%v' does not exists in params file", k)
			}
		}
	}

	network.Initialized = true
	return nil
}
