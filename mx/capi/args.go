package capi

/*
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

const MaxArgsCount = 128
const MaxCacheArgsCount = 128 * 2

var pcharCache = map[interface{}]int{}
var pcharCacheVals [MaxCacheArgsCount]struct {
	s   *C.char
	lru int64
	a   interface{}
}

var pcharCacheLru int64 = 1

func Cache(a interface{}) *C.char {
	if i, ok := pcharCache[a]; ok {
		pcharCacheVals[i].lru = pcharCacheLru
		pcharCacheLru++
		return pcharCacheVals[i].s
	}
	lru, m := pcharCacheLru, 0
	for i := 0; i < len(pcharCacheVals); i++ {
		if lru > pcharCacheVals[i].lru {
			lru = pcharCacheVals[i].lru
			m = i
		}
	}
	b := pcharCacheVals[m]
	if b.s != nil {
		delete(pcharCache, b.a)
		C.free(unsafe.Pointer(b.s))
	}
	s := C.CString(fmt.Sprintf("%v", a))
	pcharCacheVals[m].a = a
	pcharCacheVals[m].s = s
	pcharCacheVals[m].lru = pcharCacheLru
	pcharCache[a] = m
	pcharCacheLru++
	return s
}

func Fillargs(keys []*C.char, vals []*C.char, ap []interface{}) int {
	i := 0
	for len(ap) != 0 && i < len(vals) {
		k := ap[0].(MxnetKey)
		if k != KeyEmpty {
			keys[i] = mxkeys[k]
			vals[i] = Cache(ap[1])
			i++
		}
		ap = ap[2:]
	}
	return i
}
