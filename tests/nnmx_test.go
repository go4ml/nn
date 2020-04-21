package tests

import (
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/nn/mx"
	"gotest.tools/assert"
	"reflect"
	"testing"
)

var test_on_GPU = mx.GpuCount() > 0 && false
var RequiredMxVersion = fu.MakeVersion(1, 5, 0)

func Test_0info(t *testing.T) {
	t.Logf("go-dlnn version %v", mx.Version)
	t.Logf("libmxnet version %v", mx.LibVersion())
	t.Logf("GPUs count %v", mx.GpuCount())
	s := ""
	if !test_on_GPU {
		s = " not"
	}
	t.Logf("will%s test on GPU", s)
}

func Test_Version(t *testing.T) {
	assert.Assert(t, RequiredMxVersion == mx.Version)
}

func assertPanic(t *testing.T, f func()) {
	defer func() {
		assert.Assert(t, recover() != nil, "The code did not panic")
	}()
	f()
}

func f_Array1(ctx mx.Context, t *testing.T) {
	t.Logf("Array on %v", ctx)
	a := ctx.Array(mx.Int32, mx.Dim(100))
	defer a.Release()
	assert.Assert(t, a != nil)
	assert.Assert(t, a.Dtype().String() == "Int32")
	assert.Assert(t, a.Dim().String() == "(100)")
	b := mx.Array(mx.Float32, mx.Dim(100, 10))
	defer b.Release()
	assert.Assert(t, b != nil)
	assert.Assert(t, b.Dtype().String() == "Float32")
	assert.Assert(t, b.Dim().String() == "(100,10)")
	c := mx.Array(mx.Uint8, mx.Dim(100, 10, 1))
	defer c.Release()
	assert.Assert(t, c.Dtype().String() == "Uint8")
	assert.Assert(t, c.Dim().String() == "(100,10,1)")
	assert.Assert(t, PanicWith("failed to create array", func() {
		_ = mx.Array(mx.Int64, mx.Dim(100000000000, 100000000, 1000000000))
	}))
	assert.Assert(t, PanicWith("bad dimension", func() {
		_ = mx.Array(mx.Int64, mx.Dim())
	}))
	assert.Assert(t, PanicWith("bad dimension", func() {
		_ = mx.Array(mx.Int64, mx.Dim(-1, 3))
	}))
	assert.Assert(t, PanicWith("bad dimension", func() {
		_ = mx.Array(mx.Int64, mx.Dim(1, 3, 10, 100, 2))
	}))
}

func Test_Array1(t *testing.T) {
	f_Array1(mx.CPU, t)
	if test_on_GPU {
		f_Array1(mx.GPU0, t)
	}
}

var dtypeMap = map[mx.Dtype]reflect.Type{
	mx.Float64: reflect.TypeOf(float64(0)),
	mx.Float32: reflect.TypeOf(float32(0)),
	mx.Float16: reflect.TypeOf(float32(0)),
	mx.Int8:    reflect.TypeOf(int8(0)),
	mx.Uint8:   reflect.TypeOf(uint8(0)),
	mx.Int32:   reflect.TypeOf(int32(0)),
	mx.Int64:   reflect.TypeOf(int64(0)),
}

type array2_ds_t struct {
	mx.Dtype
	vals interface{}
}

var array2_ds = []array2_ds_t{
	{mx.Float32, []interface{}{.1, int(2), int64(3), float64(4), .0005}},
	{mx.Int32, []interface{}{.1, int(2), int64(3), float64(4), .0005}},
	{mx.Uint8, []int{1, 2, 3, 4, 5}},
	{mx.Int8, []int{1, 2, 3, 4, 5}},
	{mx.Int64, []int{1, 2, 3, 4, 5}},
	// Float16 has a round error so using integer values only
	{mx.Float16, []interface{}{1, int(2), int64(3), float64(4), 5}},
}

func compare(t *testing.T, data, result interface{}, dt mx.Dtype, no int) bool {
	v0 := reflect.ValueOf(data)
	if v0.Kind() != reflect.Slice && v0.Kind() != reflect.Array {
		t.Errorf("test data is not slice")
		return false
	}
	v1 := reflect.ValueOf(result)
	if v1.Kind() != reflect.Slice && v1.Kind() != reflect.Array {
		t.Errorf("test result is not slice")
		return false
	}
	if v0.Len() > v1.Len() {
		t.Errorf("test data is longer than test result")
		return false
	}
	for i := 0; i < v0.Len(); i++ {
		q := func(v reflect.Value) reflect.Value {
			if v.Kind() == reflect.Interface {
				v = v.Elem()
			}
			return v.Convert(dtypeMap[dt])
		}
		val0 := q(v0.Index(i))
		val1 := q(v1.Index(i))
		if !reflect.DeepEqual(val0.Interface(), val1.Interface()) {
			t.Errorf("%v at %v: %v != %v, %v", no, i, val0, val1, dt)
			return false
		}
	}
	return true
}

func f_Array2(ctx mx.Context, t *testing.T) {
	t.Logf("Array on %v", ctx)
	for no, v := range array2_ds {
		a := ctx.Array(v.Dtype, mx.Dim(5), v.vals)
		defer a.Release()
		if v.Dtype == mx.Float16 && ctx.IsGPU() {
			t.Logf("Float16 is unsupported on GPU, skipped")
			continue // can be unsupported on GPU
		}
		dt := v.Dtype
		if dt == mx.Float16 {
			dt = mx.Float32
		}
		vals := a.Values(dt)
		assert.Check(t, compare(t, v.vals, vals, dt, no))
	}
}

func Test_Array2(t *testing.T) {
	f_Array2(mx.CPU, t)
	if test_on_GPU {
		f_Array2(mx.GPU0, t)
	}
}

func Test_Array3(t *testing.T) {
	var err error
	ds := []int{1, 2, 3, 4, 5}
	a := mx.CPU.Array(mx.Float16, mx.Dim(5), 1, 2, 3, 4, 5)
	defer a.Release()
	assert.Assert(t, PanicWith("Float16", func() {
		_ = a.Values(mx.Float16)
	}))
	v := a.Values(mx.Float32)
	assert.NilError(t, err)
	assert.Check(t, compare(t, ds, v, mx.Float32, 0))
}

func f_Random(ctx mx.Context, t *testing.T) {
	t.Logf("Random_Uniform on %v", ctx)
	a := ctx.Array(mx.Float32, mx.Dim(1, 3)).Uniform(0, 1)
	defer a.Release()
	t.Log(a.ValuesF32())
}

func Test_Random(t *testing.T) {
	mx.RandomSeed(42)
	f_Random(mx.CPU, t)
	if test_on_GPU {
		mx.RandomSeed(42)
		f_Random(mx.GPU0, t)
	}
}

func f_Xavier(ctx mx.Context, t *testing.T) {
	t.Logf("Random_Uniform on %v", ctx)
	a := ctx.Array(mx.Float32, mx.Dim(1, 3)).Xavier(false, 2, 3)
	defer a.Release()
	t.Log(a.ValuesF32())
}

func Test_Xavier(t *testing.T) {
	mx.RandomSeed(42)
	f_Xavier(mx.CPU, t)
	if test_on_GPU {
		mx.RandomSeed(42)
		f_Xavier(mx.GPU0, t)
	}
}

func f_Zeros(ctx mx.Context, t *testing.T) {
	t.Logf("Zeros on %v", ctx)
	a := ctx.Array(mx.Float32, mx.Dim(1, 3)).Zeros()
	defer a.Release()
	assert.Assert(t, compare(t, a.ValuesF32(), []float32{0, 0, 0}, mx.Float32, 0))
}

func Test_Zeros(t *testing.T) {
	f_Zeros(mx.CPU, t)
	if test_on_GPU {
		f_Zeros(mx.GPU0, t)
	}
}

func Test_SetValues(t *testing.T) {
	a := mx.CPU.Array(mx.Float32, mx.Dim(2, 3))
	defer a.Release()
	assert.Assert(t, PanicWith("not enough", func() {
		a.SetValues([]float32{1, 2, 3})
	}))
	assert.Assert(t, PanicWith("too many", func() {
		a.SetValues([]float32{1, 2, 3, 4, 5, 6, 7})
	}))
	assert.Assert(t, PanicWith("not enough", func() {
		a.SetValues([][]float32{{1, 2, 3, 4}})
	}))
	assert.Assert(t, PanicWith("too many", func() {
		a.SetValues([][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
	}))
	a.SetValues([][]float32{{2, 3, 4}, {5, 6, 7}})
	assert.Assert(t, compare(t, a.ValuesF32(), []float32{2, 3, 4, 5, 6, 7}, mx.Float32, 0))
	a.SetValues([]float32{9, 8, 7, 6, 5, 4})
	assert.Assert(t, compare(t, a.ValuesF32(), []float32{9, 8, 7, 6, 5, 4}, mx.Float32, 0))
}
