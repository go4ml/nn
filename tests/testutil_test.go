package tests

import (
	"fmt"
	"gotest.tools/assert/cmp"
	"strings"
)

func PanicWith(text string, f func()) cmp.Comparison {
	return func() (result cmp.Result) {
		defer func() {
			if err := recover(); err != nil {
				s := fmt.Sprint(err)
				if strings.Index(s, text) >= 0 {
					result = cmp.ResultSuccess
					return
				}
				result = cmp.ResultFailure("panic `" + s + "` does not contain `" + text + "`")
			}
		}()
		f()
		return cmp.ResultFailure("did not panic")
	}

}
