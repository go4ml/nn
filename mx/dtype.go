package mx

type Dtype int

const (
	Float32 Dtype = 0
	Float64 Dtype = 1
	Float16 Dtype = 2
	Uint8   Dtype = 3
	Int32   Dtype = 4
	Int8    Dtype = 5
	Int64   Dtype = 6
)

func (tp Dtype) String() string {
	switch tp {
	case Float32:
		return "Float32"
	case Float64:
		return "Float"
	case Float16:
		return "Float16"
	case Uint8:
		return "Uint8"
	case Int32:
		return "Int32"
	case Int8:
		return "Int8"
	case Int64:
		return "Int64"
	default:
		panic("bad type")
	}
}

func (tp Dtype) Size() int {
	switch tp {
	case Float32:
		return 4
	case Float64:
		return 8
	case Float16:
		return 2
	case Uint8:
		return 1
	case Int32:
		return 4
	case Int8:
		return 1
	case Int64:
		return 8
	default:
		panic("bad type")
	}
}
