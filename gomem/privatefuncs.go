package gomem

import "errors"

//This contains the private funcs for HArray3d append

func (ray *HArray3d) appendhrayfloat32(input []float32) error {
	if len(input)%(ray.X*ray.Y) != 0 {
		return errors.New("[]float32 array not in multiples of ray.X*ray.Y")
	}
	zsize := len(input) / (ray.X * ray.Y)
	z := ray.Z
	ray.Z += zsize
	newsize := ray.X * ray.Y * ray.Z
	newarray := make([]float32, newsize)

	for i := 0; i < ray.X; i++ {
		for j := 0; j < ray.Y; j++ {
			for k := 0; k < z; k++ {
				newarray[(ray.Y*ray.Z*i)+(ray.Z*j)+k] = ray.Data[(ray.Y*ray.Z*i)+(ray.Z*j)+k]
			}
			for k := z; k < ray.Z; k++ {
				newarray[(ray.Y*ray.Z*i)+(ray.Z*j)+k] = input[(ray.Y*ray.Z*i)+(ray.Z*j)+k]
			}

		}
	}
	ray.Data = newarray
	return nil

}
func (ray *HArray3d) appendHArray3d(input HArray3d) error {
	if ray.X != input.X || ray.Y != input.Y {
		return errors.New("X,Y corordinates don't raych")
	}
	z := ray.Z
	ray.Z += input.Z
	newsize := ray.X * ray.Y * ray.Z
	newarray := make([]float32, newsize)

	for i := 0; i < ray.X; i++ {
		for j := 0; j < ray.Y; j++ {
			for k := 0; k < z; k++ {
				newarray[(ray.Y*ray.Z*i)+(ray.Z*j)+k] = ray.Data[(ray.Y*ray.Z*i)+(ray.Z*j)+k]
			}
			for k := z; k < ray.Z; k++ {
				newarray[(ray.Y*ray.Z*i)+(ray.Z*j)+k] = input.Data[(ray.Y*ray.Z*i)+(ray.Z*j)+k]
			}

		}
	}
	ray.Data = newarray
	return nil
}
func (ray *HArray3d) appendHArray2d(input HArray2d) error {
	if ray.X != input.X || ray.Y != input.Y {
		return errors.New("X,Y corordinates don't raych")
	}
	z := ray.Z
	ray.Z++
	newsize := ray.X * ray.Y * ray.Z
	newarray := make([]float32, newsize)

	for i := 0; i < ray.X; i++ {
		for j := 0; j < ray.Y; j++ {
			for k := 0; k < z; k++ {
				newarray[(ray.Y*ray.Z*i)+(ray.Z*j)+k] = ray.Data[(ray.Y*ray.Z*i)+(ray.Z*j)+k]
			}
			newarray[(ray.Y*ray.Z*i)+(ray.Z*j)+z] = input.Data[(ray.Y*i)+j]
		}
	}
	ray.Data = newarray
	return nil
}

func (ray *HArray3d) copydata() []float32 {
	copy := make([]float32, len(ray.Data))
	for i := 0; i < len(copy); i++ {
		copy[i] = ray.Data[i]
	}
	return copy
}
