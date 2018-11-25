package reduce

import gocudnn "github.com/dereklstinson/GoCudnn"

//OpMode is the flag to designate what mode the reduce op will do.
type OpMode gocudnn.ReduceTensorOp

func (o OpMode) cu() gocudnn.ReduceTensorOp {
	return gocudnn.ReduceTensorOp(o)
}

//OpFlags passes OpMode Flags
type OpFlags struct {
	f gocudnn.ReduceTensorOpFlag
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
