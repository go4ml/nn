package capi

/*
#include "capi.h"
*/
import "C"

import (
	"fmt"
	"go-ml.dev/pkg/dyl"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/zorros/zlog"
	"runtime"
	"unsafe"
)

var GpuCount int = 0
var LibVersion = 0
var mxkeys = map[MxnetKey]*C.char{}
var mxentry = map[MxnetOp]C.AtomicSymbolCreator{}

type NDArrayHandle = C.NDArrayHandle
type SymbolHandle = C.SymbolHandle
type ExecutorHandle = C.ExecutorHandle

func ReleaseNDArry(handle NDArrayHandle) {
	if handle != nil {
		C.MXNDArrayFree(handle)
	}
}

func ReleaseSymbol(handle SymbolHandle) {
	if handle != nil {
		C.MXSymbolFree(handle)
	}
}

func ReleaseExecutor(handle ExecutorHandle) {
	if handle != nil {
		C.MXExecutorFree(handle)
	}
}

func mxLastError() string {
	e := C.MXGetLastError()
	return C.GoString(e)
}

func init() {

	var so dyl.SO
	dlVerbose := dyl.Verbose(func(text string, verbosity int) {
		if verbosity < 2 {
			// verbosity 2 relates to detailed information
			fmt.Println(text)
		}
	})
	if runtime.GOOS == "linux" && runtime.GOARCH == "amd64" {
		so = dyl.Load(
			dyl.Custom("/opt/mxnet/lib/libmxnet.so"),
			dyl.Cached("dl/go-model/libmxnet.so"),
			dyl.System("libmxnet.so"),
			dyl.LzmaExternal("https://github.com/sudachen/mxnet/releases/download/1.5.0-mkldnn-static/libmxnet_cpu_lin64.lzma"),
			dlVerbose)
	} else if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		so = dyl.Load(
			dyl.Cached("dl/go-model/mxnet15.dll"),
			dyl.System("mxnet15.dll"),
			dyl.LzmaExternal("https://github.com/sudachen/mxnet/releases/download/1.5.0-mkldnn-static/libmxnet_cpu_win64.lzma"),
			dlVerbose)
	} else {
		panic("unsupported platform")
	}

	so.Bind("MXGetVersion", unsafe.Pointer(&C._godl_MXGetVersion))
	so.Bind("MXGetLastError", unsafe.Pointer(&C._godl_MXGetLastError))
	so.Bind("MXGetGPUCount", unsafe.Pointer(&C._godl_MXGetGPUCount))
	so.Bind("MXNDArrayCreateEx", unsafe.Pointer(&C._godl_MXNDArrayCreateEx))
	so.Bind("MXNDArrayFree", unsafe.Pointer(&C._godl_MXNDArrayFree))
	so.Bind("MXNDArrayGetDType", unsafe.Pointer(&C._godl_MXNDArrayGetDType))
	so.Bind("MXNDArrayGetShape", unsafe.Pointer(&C._godl_MXNDArrayGetShape))
	so.Bind("MXNDArraySyncCopyFromCPU", unsafe.Pointer(&C._godl_MXNDArraySyncCopyFromCPU))
	so.Bind("MXNDArraySyncCopyToCPU", unsafe.Pointer(&C._godl_MXNDArraySyncCopyToCPU))
	so.Bind("MXExecutorBackward", unsafe.Pointer(&C._godl_MXExecutorBackward))
	so.Bind("MXExecutorForward", unsafe.Pointer(&C._godl_MXExecutorForward))
	so.Bind("MXExecutorBind", unsafe.Pointer(&C._godl_MXExecutorBind))
	so.Bind("MXExecutorFree", unsafe.Pointer(&C._godl_MXExecutorFree))
	so.Bind("MXExecutorOutputs", unsafe.Pointer(&C._godl_MXExecutorOutputs))
	so.Bind("MXRandomSeed", unsafe.Pointer(&C._godl_MXRandomSeed))
	so.Bind("MXRandomSeedContext", unsafe.Pointer(&C._godl_MXRandomSeedContext))
	so.Bind("MXSymbolCompose", unsafe.Pointer(&C._godl_MXSymbolCompose))
	so.Bind("MXSymbolCreateAtomicSymbol", unsafe.Pointer(&C._godl_MXSymbolCreateAtomicSymbol))
	so.Bind("MXSymbolCreateGroup", unsafe.Pointer(&C._godl_MXSymbolCreateGroup))
	so.Bind("MXSymbolCreateVariable", unsafe.Pointer(&C._godl_MXSymbolCreateVariable))
	so.Bind("MXSymbolFree", unsafe.Pointer(&C._godl_MXSymbolFree))
	so.Bind("MXSymbolGetAtomicSymbolName", unsafe.Pointer(&C._godl_MXSymbolGetAtomicSymbolName))
	so.Bind("MXSymbolGetInternals", unsafe.Pointer(&C._godl_MXSymbolGetInternals))
	so.Bind("MXSymbolInferShape", unsafe.Pointer(&C._godl_MXSymbolInferShape))
	so.Bind("MXSymbolListArguments", unsafe.Pointer(&C._godl_MXSymbolListArguments))
	so.Bind("MXSymbolListAtomicSymbolCreators", unsafe.Pointer(&C._godl_MXSymbolListAtomicSymbolCreators))
	so.Bind("MXSymbolListAuxiliaryStates", unsafe.Pointer(&C._godl_MXSymbolListAuxiliaryStates))
	so.Bind("MXSymbolListOutputs", unsafe.Pointer(&C._godl_MXSymbolListOutputs))
	so.Bind("MXSymbolSaveToJSON", unsafe.Pointer(&C._godl_MXSymbolSaveToJSON))
	so.Bind("MXImperativeInvoke", unsafe.Pointer(&C._godl_MXImperativeInvoke))

	var v C.int
	C.MXGetVersion(&v)
	LibVersion = int(v)

	var c C.int
	C.MXGetGPUCount(&c)
	GpuCount = int(c)

	for i := KeyEmpty + 1; i < KeyNoKey; i++ {
		mxkeys[i] = C.CString(i.Value())
	}

	var ascv *C.AtomicSymbolCreator
	var ascn C.uint

	if e := C.MXSymbolListAtomicSymbolCreators(&ascn, &ascv); e != 0 {
		panic("failed to gather symbols from mxnet")
	}

	m := map[string]MxnetOp{}
	for op := OpEmpty + 1; op < OpNoOp; op++ {
		m[op.Value()] = op
	}

	for i := 0; i < int(ascn); i++ {
		a := *(*C.AtomicSymbolCreator)(fu.Index(i, ascv))
		var n *C.char
		if e := C.MXSymbolGetAtomicSymbolName(a, &n); e != 0 {
			panic(fmt.Sprintf("failed to gather name for symbol %x", a))
		}
		if ent, ok := m[C.GoString(n)]; ok {
			mxentry[ent] = a
		}
	}

	notInit := false
	for n,v := range opmap {
		if _, ok := mxentry[n]; !ok {
			notInit = true
			//panic(fmt.Sprintf("mxnet operator %v is not found in shared library",v))
			fmt.Printf("mxnet operator %v is not found in shared library\n",v)
		}
	}

	if notInit {
		zlog.Infof("available operators:")
		for i := 0; i < int(ascn); i++ {
			a := *(*C.AtomicSymbolCreator)(fu.Index(i, ascv))
			var n *C.char
			if e := C.MXSymbolGetAtomicSymbolName(a, &n); e != 0 {
				panic(fmt.Sprintf("failed to gather name for symbol %x", a))
			}
			zlog.Info(C.GoString(n))
		}
		panic("not initialized")
	}
}

func ImperativeInvokeInplace1(op MxnetOp, h NDArrayHandle, a ...interface{}) {
	if h == nil {
		panic("uninitialized or broken array")
	}

	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char
	ano := C.int(Fillargs(keys[:], vals[:], a))

	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invoke1_inplace(ent, C.NDArrayHandle(h), ano, &keys[0], &vals[0]); e != 0 {
			panic(fmt.Sprintf("maxnet %v error: %v", op.Value(), mxLastError()))
		}
	} else {
		panic(fmt.Sprintf("unresolved API entry %v", op.Value()))
	}
}

func ImperativeInvokeInOut1(op MxnetOp, h NDArrayHandle, o NDArrayHandle, a ...interface{}) {
	if h == nil {
		panic("uninitialized or broken input array")
	}
	if o == nil {
		panic("uninitialized or broken output array")
	}

	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char
	ano := C.int(Fillargs(keys[:], vals[:], a))
	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invoke1_inout(ent, C.NDArrayHandle(h), C.NDArrayHandle(o), ano, &keys[0], &vals[0]); e != 0 {
			panic(fmt.Sprintf("maxnet %v error: %v", op.Value(), mxLastError()))
		}
	} else {
		panic(fmt.Sprintf("unresolved API entry %v", op.Value()))
	}
}

func NewNDArrayHandle(devType int, devNo int, dtype int, shape [4]int, slen int) NDArrayHandle {
	var a C.NDArrayHandle
	s := [4]C.uint{C.uint(shape[0]), C.uint(shape[1]), C.uint(shape[2]), C.uint(shape[3])}
	if e := C.MXNDArrayCreateEx(&s[0], C.uint(slen), C.int(devType), C.int(devNo), 0, C.int(dtype), &a); e != 0 {
		panic(fmt.Sprintf("failed to create array: %v", mxLastError()))
	}
	return NDArrayHandle(a)
}

func GetNDArrayRawData(handle NDArrayHandle, p unsafe.Pointer, len int) {
	if handle != nil {
		if e := C.MXNDArraySyncCopyToCPU(handle, p, C.ulong(len)); e != 0 {
			panic(fmt.Sprintf("failed to get raw data: %v", mxLastError()))
		}
	}
}

func SetNDArrayRawData(handle NDArrayHandle, p unsafe.Pointer, len int) {
	if handle != nil {
		if e := C.MXNDArraySyncCopyFromCPU(handle, p, C.ulong(len)); e != 0 {
			panic(fmt.Sprintf("failed to set raw data: %v", mxLastError()))
		}
	}
}

func CreateVariable(name string) SymbolHandle {
	var r SymbolHandle
	str := C.CString(name)
	defer C.free(unsafe.Pointer(str))
	if e := C.MXSymbolCreateVariable(str, &r); e != 0 {
		panic(fmt.Sprintf("failed to create symbolic variable: %v", mxLastError()))
	}
	return r
}

func NewSymbol(op MxnetOp, attr map[MxnetKey]string, a ...interface{}) SymbolHandle {

	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char

	if len(a)+len(attr) >= MaxArgsCount {
		panic(fmt.Sprintf("number of keys and vals must be less than %v", MaxArgsCount))
	}

	i := Fillargs(keys[:], vals[:], a)
	if attr != nil {
		for k, v := range attr {
			keys[i] = mxkeys[k]
			vals[i] = Cache(v)
			i++
		}
	}
	ano := C.int(len(a)/2 + len(attr))

	var h SymbolHandle
	if ent := mxentry[op]; ent != nil {
		if e := C.MXSymbolCreateAtomicSymbol(ent, C.uint(ano), &keys[0], &vals[0], &h); e != 0 {
			panic(fmt.Sprintf("failed to create mxnet symbol %v: %v", op.Value(), mxLastError()))
		}
	} else {
		panic(fmt.Sprintf("unresolved API entry %v", op.Value()))
	}

	return h
}

func ComposeSymbol(handle SymbolHandle, name string, a ...SymbolHandle) {
	str := C.CString(name)
	defer C.free(unsafe.Pointer(str))
	if len(a) > 0 {
		if e := C.MXSymbolCompose(handle, str, C.uint(len(a)), nil, &a[0]); e != 0 {
			panic(fmt.Sprintf("failed to compose mxnet symbol %v: %v", name, mxLastError()))
		}
	} else {
		if e := C.MXSymbolCompose(handle, str, C.uint(0), nil, nil); e != 0 {
			panic(fmt.Sprintf("failed to compose mxnet symbol %v: %v", name, mxLastError()))
		}
	}
}

const ArgumentsNames = 0
const OutputNames = 1
const AuxNames = 2

func ListNames(handle SymbolHandle, kind int) []string {
	var (
		i      int
		e      C.int
		out_nn C.uint
		out_ns **C.char
		r      []string
	)

	switch kind {
	case ArgumentsNames:
		e = C.MXSymbolListArguments(
			handle,
			&out_nn,
			&out_ns)
	case OutputNames:
		e = C.MXSymbolListOutputs(
			handle,
			&out_nn,
			&out_ns)
	case AuxNames:
		e = C.MXSymbolListAuxiliaryStates(
			handle,
			&out_nn,
			&out_ns)
	}

	if e != 0 {
		panic(fmt.Sprintf("failed to request output names from mxnet: %v", mxLastError()))
	}

	name_at := func(i int) string {
		return C.GoString(*(**C.char)(fu.Index(i, out_ns)))
	}

	r = make([]string, int(out_nn))

	for i = 0; i < int(out_nn); i++ {
		r[i] = name_at(i)
	}

	return r
}

const WithArguments = 1
const WithOutputs = 2
const WithAuxStates = 4
const WithoutOutput = 8

func InferShapes(handle SymbolHandle, with map[string][]int, selector int) map[string][]int {

	if len(with) > MaxArgsCount {
		panic("to many shapes in args")
	}

	var (
		keys                  [MaxArgsCount]*C.char
		si                    [MaxArgsCount]C.uint
		sd                    [MaxArgsCount * 4]C.uint
		in_ss, out_ss, aux_ss C.uint
		in_sn, out_sn, aux_sn *C.uint
		in_sd, out_sd, aux_sd **C.uint
		complete              C.int
	)

	i, j := 0, 0
	for s, v := range with {
		keys[i] = C.CString(s)
		i++
		for _, t := range v {
			sd[j] = C.uint(t)
			j++
		}
		si[i] = C.uint(j)
	}

	defer func() {
		for _, p := range keys {
			C.free(unsafe.Pointer(p))
		}
	}()

	e := C.MXSymbolInferShape(
		handle,
		C.uint(len(with)),
		&keys[0],
		&si[0],
		&sd[0],
		&in_ss, &in_sn, &in_sd,
		&out_ss, &out_sn, &out_sd,
		&aux_ss, &aux_sn, &aux_sd,
		&complete)
	if e != 0 {
		panic(fmt.Sprintf("failed to request shapes from mxnet: %v", mxLastError()))
	}

	shape_at := func(i int, d *C.uint, s **C.uint) []int {
		n := int(*(*C.uint)(fu.Index(i, d)))
		r := make([]int, n)
		ps := *(**C.uint)(fu.Index(i, s))

		for j := 0; j < n; j++ {
			r[j] = int(*(*C.int)(fu.Index(j, ps)))
		}

		return r
	}

	r := make(map[string][]int)

	if (selector & WithArguments) != 0 {
		names := ListNames(handle, 0)
		for i, name := range names {
			r[name] = shape_at(i, in_sn, in_sd)
		}
	}

	if (selector & WithOutputs) != 0 {
		names := ListNames(handle, 1)

		for i, name := range names {
			if _, ok := with[name]; !ok {
				r[name] = shape_at(i, out_sn, out_sd)
			}
		}
	}

	if (selector & WithAuxStates) != 0 {
		names := ListNames(handle, 2)

		for i, name := range names {
			if _, ok := with[name]; !ok {
				r[name] = shape_at(i, aux_sn, aux_sd)
			}
		}
	}

	if (selector & WithoutOutput) == 0 {
		r["_0"] = shape_at(0, out_sn, out_sd)
	}

	return r
}

func GroupSymbols(s []SymbolHandle) SymbolHandle {
	var r SymbolHandle
	e := C.MXSymbolCreateGroup(C.uint(len(s)), &s[0], &r)
	if e != 0 {
		panic(fmt.Sprintf("failed to group mxnet symbols: %v", mxLastError()))
	}
	return r
}

func GetInternals(s SymbolHandle) SymbolHandle {
	var o SymbolHandle
	if e := C.MXSymbolGetInternals(s, &o); e != 0 {
		panic(fmt.Sprintf("failed to get mxnet symbol internals: %v", mxLastError()))
	}
	return o
}

func Bind(symbol SymbolHandle, devType, devNo int, args []NDArrayHandle, grads []NDArrayHandle, aux []NDArrayHandle) ExecutorHandle {
	var r ExecutorHandle

	ga := make([]C.uint, len(args))
	for i := range ga {
		if grads[i] != nil {
			ga[i] = 1
		}
	}

	paux := aux
	if len(aux) == 0 {
		paux = []NDArrayHandle{nil}[:]
	}

	e := C.MXExecutorBind(
		symbol,
		C.int(devType),
		C.int(devNo),
		C.uint(len(args)),
		&args[0],
		&grads[0],
		&ga[0],
		C.uint(len(aux)),
		&paux[0],
		&r)

	if e != 0 {
		panic("failed to bind mxnet symbols: " + mxLastError())
	}

	return r
}

type NDArrayInfo struct {
	Handle NDArrayHandle
	Dim    []int
	Type   int
}

func FillInfo(nfo *NDArrayInfo) {
	var (
		dt C.int
		dn C.uint
		ds *C.uint
	)
	if e := C.MXNDArrayGetDType(nfo.Handle, &dt); e != 0 {
		panic("failed to get dtype of mxnet ndarray: " + mxLastError())
	}
	nfo.Type = int(dt)
	if e := C.MXNDArrayGetShape(nfo.Handle, &dn, &ds); e != 0 {
		panic("failed to get shape of mxnet ndarray: " + mxLastError())
	}
	nfo.Dim = make([]int, int(dn))
	for i := range nfo.Dim {
		nfo.Dim[i] = int(*(*C.int)(fu.Index(i, ds)))
	}
}

func GetOutputs(exec ExecutorHandle) []NDArrayInfo {
	var (
		n C.uint
		a *NDArrayHandle
		e C.int
	)
	if e = C.MXExecutorOutputs(exec, &n, &a); e != 0 {
		panic("failed get mxnet outputs: " + mxLastError())
	}
	r := make([]NDArrayInfo, int(n))
	for i := range r {
		r[i].Handle = *(*NDArrayHandle)(fu.Index(i, a))
		FillInfo(&r[i])
	}
	return r
}

func Forward(exec ExecutorHandle, train bool) {
	t := C.int(0)
	if train {
		t = C.int(1)
	}
	if e := C.MXExecutorForward(exec, t); e != 0 {
		panic("failed on mxnet forward: " + mxLastError())
	}
}

func Backward(exec ExecutorHandle) {
	if e := C.MXExecutorBackward(exec, C.uint(0), nil); e != 0 {
		panic("failed on mxnet backward: " + mxLastError())
	}
}

func OptimizerUpdate(op MxnetOp, params, grads, state1 NDArrayHandle, state2 NDArrayHandle, a ...interface{}) {
	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char
	ano := C.int(Fillargs(keys[:], vals[:], a))
	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invokeN_inout(ent, params, ano, &keys[0], &vals[0], params, grads, state1, state2); e != 0 {
			panic(fmt.Sprintf("mxnet api %v error: %v", op.Value(), mxLastError()))
		}
	} else {
		panic("unresolved API entry " + op.Value())
	}
}

func ToJson(sym SymbolHandle) []byte {
	var s *C.char
	if e := C.MXSymbolSaveToJSON(sym, &s); e != 0 {
		panic("mxnet failed to stringify symbol: " + mxLastError())
	}
	ln := int(C.strlen(s))
	bs := make([]byte, ln)
	C.memcpy(unsafe.Pointer(&bs[0]), unsafe.Pointer(s), C.ulong(ln))
	return bs
}

func RandomSeed(seed int) {
	if e := C.MXRandomSeed(C.int(seed)); e != 0 {
		panic("mxnet failed to set ramdom seed: " + mxLastError())
	}
}

func ContextRandomSeed(seed, devType, devNo int) {
	if e := C.MXRandomSeedContext(C.int(seed), C.int(devType), C.int(devNo)); e != 0 {
		panic("mxnet failed to set ramdom seed: " + mxLastError())
	}
}
