package reduce

import (
	"strings"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//OpMode is the flag to designate what mode the reduce op will do.
type OpMode gocudnn.ReduceTensorOp

func (r OpMode) cu() gocudnn.ReduceTensorOp {
	return gocudnn.ReduceTensorOp(r)
}

//OpFlags passes OpMode Flags
type OpFlags struct {
	f gocudnn.ReduceTensorOpFlag
}

//ToOpMode to op mode will take a string and change it to the op mode.  If a string is not recognized then it will make the op mode set at average
func ToOpMode(c string) OpMode {
	var flgs OpFlags
	lowerc := strings.ToLower(c)
	switch lowerc {
	case "add":
		return flgs.Add()
	case "amax":
		return flgs.Amax()
	case "avg":
		return flgs.Avg()
	case "max":
		return flgs.Max()
	case "min":
		return flgs.Min()
	case "mul":
		return flgs.Mul()
	case "mulnozeros":
		return flgs.MulNoZeros()
	case "norm1":
		return flgs.Norm1()
	case "norm2":
		return flgs.Norm2()
	default:
		return flgs.Avg()

	}
}

//Readable takes the flag and makes it a string
func (r OpMode) Readable() string {
	var flgs OpFlags
	switch r {
	case flgs.Add():
		return "Add"

	case flgs.Amax():
		return "Amax"

	case flgs.Avg():
		return "Avg"

	case flgs.Max():
		return "Max"

	case flgs.Min():
		return "Min"

	case flgs.Mul():
		return "Mul"

	case flgs.MulNoZeros():
		return "MulNoZeros"

	case flgs.Norm1():
		return "Norm1"

	case flgs.Norm2():
		return "Norm2"
	}
	return ""
}

//Add returns reduceTensorAdd flag
func (r OpFlags) Add() OpMode {
	return OpMode(r.f.Add())
}

//Mul returns reduceTensorMul flag
func (r OpFlags) Mul() OpMode {
	return OpMode(r.f.Mul())
}

//Min returns reduceTensorMin flag
func (r OpFlags) Min() OpMode {
	return OpMode(r.f.Min())
}

//Max returns reduceTensorMax flag
func (r OpFlags) Max() OpMode {
	return OpMode(r.f.Max())
}

//Amax returns reduceTensorAmax flag
func (r OpFlags) Amax() OpMode {
	return OpMode(r.f.Amax())
}

//Avg returns reduceTensorAvg flag
func (r OpFlags) Avg() OpMode {
	return OpMode(r.f.Avg())
}

//Norm1 returns reduceTensorNorm1 flag
func (r OpFlags) Norm1() OpMode {
	return OpMode(r.f.Norm1())
}

//Norm2 returns reduceTensorNorm2 flag
func (r OpFlags) Norm2() OpMode {
	return OpMode(r.f.Norm2())
}

//MulNoZeros returns reduceTensorMulNoZeros flag
func (r OpFlags) MulNoZeros() OpMode {
	return OpMode(r.f.MulNoZeros())
}
