package mat

import "errors"

//This contains the private funcs for Hmat3d append
func (mat *Hmat3d) appendhmatfloat32(input []float32) error {
	if len(input)%(mat.X*mat.Y) != 0 {
		return errors.New("[]float32 array not in multiples of mat.X*mat.Y")
	}
	zsize := len(input) / (mat.X * mat.Y)
	z := mat.Z
	mat.Z += zsize
	newsize := mat.X * mat.Y * mat.Z
	newarray := make([]float32, newsize)

	for i := 0; i < mat.X; i++ {
		for j := 0; j < mat.Y; j++ {
			for k := 0; k < z; k++ {
				newarray[(mat.Y*mat.Z*i)+(mat.Z*j)+k] = mat.Data[(mat.Y*mat.Z*i)+(mat.Z*j)+k]
			}
			for k := z; k < mat.Z; k++ {
				newarray[(mat.Y*mat.Z*i)+(mat.Z*j)+k] = input[(mat.Y*mat.Z*i)+(mat.Z*j)+k]
			}

		}
	}
	mat.Data = newarray
	return nil

}
func (mat *Hmat3d) appendhmat3d(input Hmat3d) error {
	if mat.X != input.X || mat.Y != input.Y {
		return errors.New("X,Y corordinates don't match")
	}
	z := mat.Z
	mat.Z += input.Z
	newsize := mat.X * mat.Y * mat.Z
	newarray := make([]float32, newsize)

	for i := 0; i < mat.X; i++ {
		for j := 0; j < mat.Y; j++ {
			for k := 0; k < z; k++ {
				newarray[(mat.Y*mat.Z*i)+(mat.Z*j)+k] = mat.Data[(mat.Y*mat.Z*i)+(mat.Z*j)+k]
			}
			for k := z; k < mat.Z; k++ {
				newarray[(mat.Y*mat.Z*i)+(mat.Z*j)+k] = input.Data[(mat.Y*mat.Z*i)+(mat.Z*j)+k]
			}

		}
	}
	mat.Data = newarray
	return nil
}
func (mat *Hmat3d) appendhmat2d(input Hmat2d) error {
	if mat.X != input.X || mat.Y != input.Y {
		return errors.New("X,Y corordinates don't match")
	}
	z := mat.Z
	mat.Z++
	newsize := mat.X * mat.Y * mat.Z
	newarray := make([]float32, newsize)

	for i := 0; i < mat.X; i++ {
		for j := 0; j < mat.Y; j++ {
			for k := 0; k < z; k++ {
				newarray[(mat.Y*mat.Z*i)+(mat.Z*j)+k] = mat.Data[(mat.Y*mat.Z*i)+(mat.Z*j)+k]
			}
			newarray[(mat.Y*mat.Z*i)+(mat.Z*j)+z] = input.Data[(mat.Y*i)+j]
		}
	}
	mat.Data = newarray
	return nil
}
