package npp

import (
	"errors"

	"github.com/dereklstinson/GoCudnn/npp"
)

func getUint8pointertype(src interface{}) *npp.Uint8 {
	switch x := src.(type) {
	case *npp.Uint8:
		return x

	default:
		return nil
	}
}
func getUint16pointertype(src interface{}) *npp.Uint16 {
	switch x := src.(type) {
	case *npp.Uint16:
		return x

	default:
		return nil
	}
}
func getInt16pointertype(src interface{}) *npp.Int16 {
	switch x := src.(type) {
	case *npp.Int16:
		return x

	default:
		return nil
	}
}
func getFloat32pointertype(src interface{}) *npp.Float32 {
	switch x := src.(type) {
	case *npp.Float32:
		return x

	default:
		return nil
	}
}
func getFloat64pointertype(src interface{}) *npp.Float64 {
	switch x := src.(type) {
	case *npp.Float64:
		return x

	default:
		return nil
	}
}

func ResizeSqPixelC1R(src interface{}, srcsize npp.Size, srcstep int32, srcROI npp.Rect, dst interface{}, dststep int32, dstROI npp.Rect, xFactor, yFactor, xShift, yShift float64, interpol npp.InterpolationMode) error {
	switch x := src.(type) {
	case *npp.Uint8:
		y := getUint8pointertype(dst)
		if y == nil {
			return errors.New("dst is not Uint8")
		}
		return npp.ResizeSqrPixel8uC1R(x, srcsize, srcstep, srcROI, y, dststep, dstROI, xFactor, yFactor, xShift, yShift, interpol)
	case *npp.Uint16:
		y := getUint16pointertype(dst)
		if y == nil {
			return errors.New("dst is not Uint8")
		}
		return npp.ResizeSqrPixel16uC1R(x, srcsize, srcstep, srcROI, y, dststep, dstROI, xFactor, yFactor, xShift, yShift, interpol)
	case *npp.Int16:
		y := getInt16pointertype(dst)
		if y == nil {
			return errors.New("dst is not Int16")
		}
		return npp.ResizeSqrPixel16sC1R(x, srcsize, srcstep, srcROI, y, dststep, dstROI, xFactor, yFactor, xShift, yShift, interpol)
	case *npp.Float32:
		y := getFloat32pointertype(dst)
		if y == nil {
			return errors.New("dst is not Float32")
		}
		return npp.ResizeSqrPixel32fC1R(x, srcsize, srcstep, srcROI, y, dststep, dstROI, xFactor, yFactor, xShift, yShift, interpol)
	case *npp.Float64:
		y := getFloat64pointertype(dst)
		if y == nil {
			return errors.New("dst is not Float64")
		}
		return npp.ResizeSqrPixel64fC1R(x, srcsize, srcstep, srcROI, y, dststep, dstROI, xFactor, yFactor, xShift, yShift, interpol)
	}
	return errors.New("ResizeSqPixelC1R unsupported type")
}
