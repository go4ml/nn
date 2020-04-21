package mx

import (
	"fmt"
	"go-ml.dev/pkg/nn/mx/capi"
	"reflect"
	"runtime"
	"unsafe"
)

type NDArray struct {
	ctx    Context
	dim    Dimension
	dtype  Dtype
	handle capi.NDArrayHandle
}

func release(a *NDArray) {
	if a != nil {
		capi.ReleaseNDArry(a.handle)
		a.handle = nil
	}
}

func (a *NDArray) Release() {
	release(a)
}

func Array(tp Dtype, d Dimension) *NDArray {
	return CPU.Array(tp, d)
}

func (c Context) Array(tp Dtype, d Dimension, vals ...interface{}) *NDArray {
	if !d.Good() {
		panic(fmt.Sprintf("failed to create array %v%v: bad dimension", tp.String(), d.String()))
	}
	a := &NDArray{ctx: c, dim: d, dtype: tp}
	a.handle = capi.NewNDArrayHandle(c.DevType(), c.DevNo(), int(tp), d.Shape, d.Len)
	if len(vals) > 0 {
		a.SetValues(vals...)
	}
	runtime.SetFinalizer(a, release)
	return a
}

func (c Context) CopyAs(a *NDArray, dtype Dtype) *NDArray {
	if a == nil || a.handle == nil {
		panic("can't copy broken array")
	}
	b := c.Array(dtype, a.dim)
	capi.ImperativeInvokeInOut1(capi.OpCopyTo, a.handle, b.handle)
	return b
}

func (a *NDArray) NewLikeThis() *NDArray {
	return a.ctx.Array(a.dtype, a.dim)
}

func (a *NDArray) Context() Context {
	return a.ctx
}

func (a *NDArray) Dtype() Dtype {
	return a.dtype
}

func (a *NDArray) Dim() Dimension {
	return a.dim
}

func (a *NDArray) Cast(dt Dtype) *NDArray {
	return nil
}

func (a *NDArray) Reshape(dim Dimension) *NDArray {
	return nil
}

func (a *NDArray) String() string {
	return ""
}

func (a *NDArray) Depth() int {
	return a.dim.Len
}

func (a *NDArray) Len(d int) int {
	if d < 0 || d >= 3 {
		return 0
	}
	if a.dim.Len <= d {
		return 1
	}
	return a.dim.Shape[d]
}

func (a *NDArray) Size() int {
	return a.dim.SizeOf(a.dtype)
}

var typemap = map[Dtype]reflect.Type{
	Float64: reflect.TypeOf(float64(0)),
	Float32: reflect.TypeOf(float32(0)),
	Int8:    reflect.TypeOf(int8(0)),
	Uint8:   reflect.TypeOf(uint8(0)),
	Int32:   reflect.TypeOf(int32(0)),
	Int64:   reflect.TypeOf(int64(0)),
}

var rtypemap = map[reflect.Type]Dtype{
	reflect.TypeOf(float64(0)): Float64,
	reflect.TypeOf(float32(0)): Float32,
	reflect.TypeOf(int8(0)):    Int8,
	reflect.TypeOf(uint8(0)):   Uint8,
	reflect.TypeOf(int32(0)):   Int32,
	reflect.TypeOf(int64(0)):   Int64,
}

func copyTo(s reflect.Value, n int, v0 reflect.Value, dt reflect.Type) int {
	if v0.Kind() == reflect.Interface {
		v0 = v0.Elem()
	}
	if v0.Kind() == reflect.Slice || v0.Kind() == reflect.Array {
		if v0.Type() == reflect.SliceOf(dt) && s.Len()-n >= v0.Len() {
			n += reflect.Copy(s.Slice(n, s.Len()), v0)
		} else {
			for i := 0; i < v0.Len(); i++ {
				n = copyTo(s, n, v0.Index(i), dt)
			}
		}
	} else {
		switch v0.Kind() {
		case reflect.Int, reflect.Int8, reflect.Uint8, reflect.Int16, reflect.Uint16,
			reflect.Int32, reflect.Uint32, reflect.Int64, reflect.Uint64,
			reflect.Float32, reflect.Float64:
			if s.Len() <= n {
				panic("too many elements to copy")
			}
			s.Index(n).Set(v0.Convert(dt))
			n++
		default:
			panic("can't initialize with non numeric type " + v0.Type().String())
		}
	}
	return n
}

func (a *NDArray) SetValues(vals ...interface{}) {
	if a == nil || a.handle == nil {
		panic("can't initialize broken array")
	}

	if a.dtype == Float16 {
		q := CPU.CopyAs(a, Float32)
		defer q.Release()
		q.SetValues(vals...)
		capi.ImperativeInvokeInOut1(capi.OpCopyTo, q.handle, a.handle)
		return
	}

	dt, ok := typemap[a.dtype]
	if !ok {
		panic(fmt.Sprintf("initialization with dtype %v is unsupportd", a.dtype))
	}

	s := reflect.ValueOf(vals[0])

	if len(vals) != 1 || s.Type() != reflect.SliceOf(dt) || s.Len() != a.dim.Total() {
		s = reflect.MakeSlice(reflect.SliceOf(dt), a.dim.Total(), a.dim.Total())
		n := copyTo(s, 0, reflect.ValueOf(vals), dt)
		if n != a.dim.Total() {
			panic("not enough elements to set value")
		}
	}

	capi.SetNDArrayRawData(a.handle, unsafe.Pointer(s.Index(0).UnsafeAddr()), a.dim.Total())
}

func (a *NDArray) Raw() []byte {
	ln := a.dim.Total()
	bs := make([]byte, ln)
	capi.GetNDArrayRawData(a.handle, unsafe.Pointer(&bs[0]), ln)
	return bs
}

func (a *NDArray) Values(dtype Dtype) interface{} {
	if dtype == Float16 {
		panic("can't gate values in Float16 format")
	}
	q := a
	ln := q.dim.Total()
	if q.dtype != dtype {
		q = CPU.CopyAs(q, dtype)
		defer q.Release()
	}
	vals := reflect.MakeSlice(reflect.SliceOf(typemap[dtype]), ln, ln)
	capi.GetNDArrayRawData(q.handle, unsafe.Pointer(vals.Index(0).UnsafeAddr()), ln)
	return vals.Interface()
}

func (a *NDArray) ValuesF32() []float32 {
	return a.Values(Float32).([]float32)
}

func (a *NDArray) CopyValuesTo(dst interface{}) {
	q := a
	ln := q.dim.Total()
	s := reflect.ValueOf(dst)
	t, ok := rtypemap[s.Index(0).Type()]
	if !ok {
		panic("invalid destination type " + s.Type().String())
	}

	if q.dtype != t {
		q = CPU.CopyAs(q, t)
		defer q.Release()
	}
	capi.GetNDArrayRawData(q.handle, unsafe.Pointer(s.Index(0).UnsafeAddr()), ln)
}

func (a *NDArray) ReCopyValuesTo(dst interface{}) {
	q := a
	ln := q.dim.Total()
	s := reflect.ValueOf(dst)
	t, ok := rtypemap[s.Index(0).Type()]
	if !ok {
		panic("invalid destination type " + s.Type().String())
	}

	q = CPU.CopyAs(q, t)
	defer q.Release()

	capi.GetNDArrayRawData(q.handle, unsafe.Pointer(s.Index(0).UnsafeAddr()), ln)
}
