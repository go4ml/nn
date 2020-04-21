package mx

import (
	"fmt"
	"go-ml.dev/pkg/zorros"
	"strconv"
	"strings"
)

const (
	DimRow    = 0
	DimColumn = 1
	DimDepth  = 2
	DimDepth3 = 3
)

// do not change this constant
// code can assume exactly this value
const MaxDimensionCount = 4

// Array Dimension
type Dimension struct {
	Shape [MaxDimensionCount]int `yaml:"shape,flow"`
	Len   int                    `yaml:"len"`
}

func DimensionFromString(s string) (Dimension, error) {
	r := Dimension{}
	if len(s) > 0 && s != "" {
		if s[0] != '(' || s[len(s)-1] != ')' {
			return Dimension{}, zorros.Errorf("invalid dimension string")
		}
		s := s[1 : len(s)-1]
		if len(s) > 0 && s != "" {
			d := strings.Split(s, ",")
			for i, n := range d {
				v, err := strconv.ParseInt(n, 10, 32)
				if err != nil {
					return Dimension{}, zorros.Errorf("bad dimansion value; %w", err)
				}
				r.Shape[i] = int(v)
				r.Len = i + 1
			}
		}
	}
	return r, nil
}

/*func (d *Dimension) UnmarshalYAML(value *yaml.Node) error {
	if value.Tag != "!!str" {
		return fmt.Errorf("can't decode coin")
	}

	if v, err := DimensionFromString(value.Value); err != nil {
		return err
	} else {
		*d = v
	}

	return nil
}

func (d Dimension) MarshalYAML() (interface{}, error) {
	return d.String(), nil
}*/

func (dim Dimension) Skip(n int) Dimension {
	if dim.Len <= n {
		return Dim()
	}
	d := Dimension{Len: dim.Len - n}
	copy(d.Shape[0:], dim.Shape[n:])
	return d
}

func (dim Dimension) Push(i int) Dimension {
	d := Dimension{Len: dim.Len + 1}
	copy(d.Shape[1:dim.Len+1], dim.Shape[:dim.Len])
	d.Shape[0] = i
	return d
}

func (dim Dimension) Like(b Dimension) Dimension {
	d := dim
	for i, v := range dim.Slice() {
		if v == 0 {
			d.Shape[i] = b.Shape[i]
		} else if v < 0 {
			d.Shape[i] = b.Shape[-v]
		}
	}
	return d
}

func (dim Dimension) Slice() []int {
	return dim.Shape[:dim.Len]
}

// represent array dimension as string
func (dim Dimension) String() string {
	s := "(%d,%d,%d,%d"[0:(3+(dim.Len-1)*3)] + ")"
	q := ([]interface{}{dim.Shape[0], dim.Shape[1], dim.Shape[2], dim.Shape[3]})[:dim.Len]
	return fmt.Sprintf(s, q...)
}

// check array dimension
func (dim Dimension) Good() bool {
	if dim.Len <= 0 || dim.Len > MaxDimensionCount {
		return false
	}
	for _, v := range dim.Shape[:dim.Len] {
		if v <= 0 {
			return false
		}
	}
	return true
}

func (dim Dimension) Empty() bool {
	return dim.Len <= 0
}

// sizeof whole array data
func (dim Dimension) SizeOf(dt Dtype) int {
	return dt.Size() * dim.Total()
}

// total elements in the whole array
func (dim Dimension) Total() int {
	r := 1
	for i := 0; i < dim.Len; i++ {
		r *= dim.Shape[i]
	}
	return r
}

// creates new dimension object
func Dim(a ...int) Dimension {
	var dim Dimension
	if q := len(a); q > 0 && q <= MaxDimensionCount {
		dim.Len = q
		for i, v := range a {
			dim.Shape[i] = v
		}
	}
	return dim
}
