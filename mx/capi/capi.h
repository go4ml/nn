#include <stdlib.h>
#include <string.h>

typedef unsigned int mx_uint;
typedef float mx_float;
typedef int64_t dim_t;
typedef void *NDArrayHandle;
typedef void *AtomicSymbolCreator;
typedef void *SymbolHandle;
typedef void *ExecutorHandle;

int MXImperativeInvoke(AtomicSymbolCreator creator,
	int num_inputs,
	NDArrayHandle *inputs,
	int *num_outputs,
	NDArrayHandle **outputs,
	int num_params,
	const char **param_keys,
	const char **param_vals);

int NNSymbolListInputNames(SymbolHandle symbol,
	int option,
	unsigned int *out_size,
	const char ***out_str_array);
int MXNDArrayFree(NDArrayHandle);
int MXSymbolFree(SymbolHandle);
int MXExecutorFree(ExecutorHandle);
char* MXGetLastError();

int MXExecutorBackward(ExecutorHandle handle,mx_uint len,NDArrayHandle *head_grads);
int MXExecutorBind(SymbolHandle symbol_handle,
	 int dev_type,
	 int dev_id,
	 mx_uint len,
	 NDArrayHandle *in_args,
	 NDArrayHandle *arg_grad_store,
	 mx_uint *grad_req_type,
	 mx_uint aux_states_len,
	 NDArrayHandle *aux_states,
	 ExecutorHandle *out);

int MXExecutorForward(ExecutorHandle handle, int is_train);
int MXExecutorOutputs(ExecutorHandle handle,mx_uint *out_size,NDArrayHandle **out);
int MXGetGPUCount(int* out);
int MXGetVersion(int *out);

int MXNDArrayCreateEx(const mx_uint *shape,
	mx_uint ndim,
	int dev_type,
	int dev_id,
	int delay_alloc,
	int dtype,
	NDArrayHandle *out);

int MXNDArrayGetDType(NDArrayHandle handle,int *out_dtype);
int MXNDArrayGetShape(NDArrayHandle handle,
	mx_uint *out_dim,
	const mx_uint **out_pdata);
int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,const void *data,size_t size);
int MXNDArraySyncCopyToCPU(NDArrayHandle handle,void *data,size_t size);
int MXRandomSeed(int seed);
int MXRandomSeedContext(int seed, int dev_type, int dev_id);
int MXSymbolCompose(SymbolHandle sym,
	const char *name,
	mx_uint num_args,
	const char** keys,
	SymbolHandle* args);
int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
	mx_uint num_param,
	const char **keys,
	const char **vals,
	SymbolHandle *out);
int MXSymbolCreateGroup(mx_uint num_symbols,SymbolHandle *symbols,SymbolHandle *out);
int MXSymbolCreateVariable(const char *name, SymbolHandle *out);
int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,const char **name);
int MXSymbolGetInternals(SymbolHandle symbol,SymbolHandle *out);
int MXSymbolInferShape(SymbolHandle sym,
	mx_uint num_args,
	const char** keys,
	const mx_uint *arg_ind_ptr,
	const mx_uint *arg_shape_data,
	mx_uint *in_shape_size,
	const mx_uint **in_shape_ndim,
	const mx_uint ***in_shape_data,
	mx_uint *out_shape_size,
	const mx_uint **out_shape_ndim,
	const mx_uint ***out_shape_data,
	mx_uint *aux_shape_size,
	const mx_uint **aux_shape_ndim,
	const mx_uint ***aux_shape_data,
	int *complete);
int MXSymbolListArguments(SymbolHandle symbol,mx_uint *out_size,const char ***out_str_array);
int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,AtomicSymbolCreator **out_array);
int MXSymbolListAuxiliaryStates(SymbolHandle symbol,mx_uint *out_size,const char ***out_str_array);
int MXSymbolListOutputs(SymbolHandle symbol,mx_uint *out_size,const char ***out_str_array);
int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json);

static int imperative_invoke1_inplace(AtomicSymbolCreator ent, NDArrayHandle out, int ano, const char **keys, const char **vals) {
	NDArrayHandle* out1[1] = {&out};
	int nout = 1;
	int err = MXImperativeInvoke(ent, 0, NULL, &nout, &out1[0], ano, keys, vals);
	return err;
}

static int imperative_invoke1_inout(AtomicSymbolCreator ent, NDArrayHandle in, NDArrayHandle out, int ano, const char **keys, const char **vals) {
	NDArrayHandle* out1[1] = {&out};
	int nout = 1;
	int err = MXImperativeInvoke(ent, 1, &in, &nout, &out1[0], ano, keys, vals);
	return err;
}

static int imperative_invokeN_inout(
	AtomicSymbolCreator ent,
	NDArrayHandle out,
	int ano, const char **keys, const char **vals,
	NDArrayHandle in0, NDArrayHandle in1,
	NDArrayHandle in2, NDArrayHandle in3)
{
	int nin = 0;
	NDArrayHandle* out1[1] = {&out};
	NDArrayHandle inN[4] = {in0,in1,in2,in3};
	for ( ;nin<4 && inN[nin]; ++nin) {}
	int nout = 1;
	int err = MXImperativeInvoke(ent, nin, inN, &nout, &out1[0], ano, keys, vals);
	return err;
}

#define DEFINE_JUMPER(x) \
        void *_godl_##x = (void*)0; \
        __asm__(".global "#x"\n\t"#x":\n\tmovq _godl_"#x"(%rip),%rax\n\tjmp *%rax\n")

DEFINE_JUMPER(MXGetVersion);
DEFINE_JUMPER(MXGetLastError);
DEFINE_JUMPER(MXGetGPUCount);

DEFINE_JUMPER(MXNDArrayCreateEx);
DEFINE_JUMPER(MXNDArrayFree);
DEFINE_JUMPER(MXNDArrayGetDType);
DEFINE_JUMPER(MXNDArrayGetShape);
DEFINE_JUMPER(MXNDArraySyncCopyFromCPU);
DEFINE_JUMPER(MXNDArraySyncCopyToCPU);

DEFINE_JUMPER(MXExecutorBackward);
DEFINE_JUMPER(MXExecutorForward);
DEFINE_JUMPER(MXExecutorBind);
DEFINE_JUMPER(MXExecutorFree);
DEFINE_JUMPER(MXExecutorOutputs);

DEFINE_JUMPER(MXRandomSeed);
DEFINE_JUMPER(MXRandomSeedContext);

DEFINE_JUMPER(MXSymbolCompose);
DEFINE_JUMPER(MXSymbolCreateAtomicSymbol);
DEFINE_JUMPER(MXSymbolCreateGroup);
DEFINE_JUMPER(MXSymbolCreateVariable);
DEFINE_JUMPER(MXSymbolFree);
DEFINE_JUMPER(MXSymbolGetAtomicSymbolName);
DEFINE_JUMPER(MXSymbolGetInternals);
DEFINE_JUMPER(MXSymbolInferShape);
DEFINE_JUMPER(MXSymbolListArguments);
DEFINE_JUMPER(MXSymbolListAtomicSymbolCreators);
DEFINE_JUMPER(MXSymbolListAuxiliaryStates);
DEFINE_JUMPER(MXSymbolListOutputs);
DEFINE_JUMPER(MXSymbolSaveToJSON);

DEFINE_JUMPER(MXImperativeInvoke);
