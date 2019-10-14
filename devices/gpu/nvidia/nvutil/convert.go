package nvutil

import (
	"errors"
	"fmt"
	"github.com/dereklstinson/cutil"

	"github.com/dereklstinson/GoCuNets/utils"

	//"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type MemCopier interface {
	MemCopy(src, dst cutil.Mem, sib uint) error
}

//Convert will convert the data of A to the datatype of B and put it into B. A datatype == B datatype memory will be copied from A to B
func Convert(src, dest tensor.Volume, copier MemCopier, streamctx *npp.StreamContext) error {
	if !utils.CompareInt32(src.Dims(), dest.Dims()) {
		return errors.New("Convert: src.Dims != dest.Dims")
	}
	if src.DataType() == dest.DataType() {
		return copier.MemCopy(src.Memer(), dest.Memer(), src.CurrentSizeT())
	}
	var flg gocudnn.DataType

	switch dest.DataType() {
	case flg.Double():
		return convertToDouble(src, dest, streamctx)
	case flg.Float():
		return convertToSingle(src, dest, streamctx)
	case flg.Half():
		fmt.Println("Half is not supported")
	case flg.Int32():
		return convertToInt32(src, dest, streamctx)
	case flg.Int8():
		return convertToInt8(src, dest, streamctx)
	case flg.UInt8():
		return convertToUint8(src, dest, streamctx)
	}
	return errors.New("nvutil.Convert - Not supported type")
}
func convertToUint8(src, dest tensor.Volume, streamctx *npp.StreamContext) error {
	nppuint8 := npp.MakeUint8FromUnsafe(dest.Memer().Ptr())
	var flg gocudnn.DataType
	var rmode npp.RoundMode
	rmode.RndNear()
	switch src.DataType() {
	case flg.Double():
		return errors.New("Convert: Not Supported Double to Uint8")
	case flg.Half():
		return errors.New("Convert: Not Supported half to Uint8")
	case flg.Float():
		convertedsrc := npp.MakeFloat32FromUnsafe(src.Memer().Ptr())
		return npp.Convert32f8uSfs(convertedsrc, nppuint8, utils.FindVolumeInt32(dest.Dims(), nil), rmode, 1, streamctx)
	case flg.Int32():
		return errors.New("Convert: Not Supported Int32 to Uint8")
	case flg.Int8():
		return errors.New("Convert: Not Supported Int8 to Uint8")
	}
	return errors.New("Unsupported Type")
}
func convertToInt8(src, dest tensor.Volume, streamctx *npp.StreamContext) error {
	nppint8 := npp.MakeInt8FromUnsafe(dest.Memer().Ptr())
	var rmode npp.RoundMode
	rmode.RndNear()
	var flg gocudnn.DataType

	switch src.DataType() {
	case flg.Double():
		return errors.New("Convert: Not Supported Double to Int8")

	case flg.Half():
		return errors.New("Convert: Not Supported half to Int8")
	case flg.Float():
		convertedsrc := npp.MakeFloat32FromUnsafe(src.Memer().Ptr())
		return npp.Convert32f8sSfs(convertedsrc, nppint8, utils.FindVolumeInt32(dest.Dims(), nil), rmode, 1, streamctx)
	case flg.Int32():
		return errors.New("Convert: Not Supported Int32 to Int8")
	case flg.UInt8():
		return errors.New("Convert: Not Supported Uint8 to Int8")
	}
	return errors.New("Unsupported Type")
}

func convertToInt32(src, dest tensor.Volume, streamctx *npp.StreamContext) error {
	nppint32 := npp.MakeInt32FromUnsafe(dest.Memer().Ptr())
	var flg gocudnn.DataType
	var rmode npp.RoundMode
	rmode.RndNear()
	switch src.DataType() {
	case flg.Double():
		convertedsrc := npp.MakeFloat64FromUnsafe(src.Memer().Ptr())
		return npp.Convert64f32sSfs(convertedsrc, nppint32, utils.FindVolumeInt32(dest.Dims(), nil), rmode, 1, streamctx)
	case flg.Half():
		return errors.New("Convert: Not Supported half to Int32")
	case flg.Float():
		convertedsrc := npp.MakeFloat32FromUnsafe(src.Memer().Ptr())
		return npp.Convert32f32sSfs(convertedsrc, nppint32, utils.FindVolumeInt32(dest.Dims(), nil), rmode, 1, streamctx)
	case flg.Int8():

		return errors.New("Convert: Not Supported Int8 to Int32")
	case flg.UInt8():
		return errors.New("Convert: Not Supported Uint8 to Int32")
	}
	return errors.New("Unsupported Type")
}

func convertToDouble(src, dest tensor.Volume, streamctx *npp.StreamContext) error {
	flt64 := npp.MakeFloat64FromUnsafe(dest.Memer().Ptr())
	var flg gocudnn.DataType
	switch src.DataType() {
	case flg.Float():
		convertedsrc := npp.MakeFloat32FromUnsafe(src.Memer().Ptr())
		return npp.Convert32f64f(convertedsrc, flt64, utils.FindVolumeInt32(dest.Dims(), nil), streamctx)

	case flg.Int32():
		convertedsrc := npp.MakeInt32FromUnsafe(src.Memer().Ptr())
		return npp.Convert32s64f(convertedsrc, flt64, utils.FindVolumeInt32(dest.Dims(), nil), streamctx)
	case flg.Int8():
		return errors.New("Convert: Not Supported int8 to float64")
	case flg.Half():
		return errors.New("Convert: Not Supported half to float64")
	case flg.UInt8():

		return errors.New("Convert: Not Supported uint8 to float64")
	}
	return errors.New("Unsupported Type")
}
func convertToSingle(src, dest tensor.Volume, streamctx *npp.StreamContext) error {
	flt32 := npp.MakeFloat32FromUnsafe(dest.Memer().Ptr())
	var flg gocudnn.DataType
	switch src.DataType() {
	case flg.Double():
		convertedsrc := npp.MakeFloat64FromUnsafe(src.Memer().Ptr())
		return npp.Convert64f32f(convertedsrc, flt32, utils.FindVolumeInt32(dest.Dims(), nil), streamctx)
	case flg.Half():
		return errors.New("Convert: Not Supported half to float32")
	case flg.Int32():
		convertedsrc := npp.MakeInt32FromUnsafe(src.Memer().Ptr())
		return npp.Convert32s32f(convertedsrc, flt32, utils.FindVolumeInt32(dest.Dims(), nil), streamctx)
	case flg.Int8():
		convertedsrc := npp.MakeInt8FromUnsafe(src.Memer().Ptr())
		return npp.Convert8s32f(convertedsrc, flt32, utils.FindVolumeInt32(dest.Dims(), nil), streamctx)

	case flg.UInt8():
		convertedsrc := npp.MakeUint8FromUnsafe(src.Memer().Ptr())
		return npp.Convert8u32f(convertedsrc, flt32, utils.FindVolumeInt32(dest.Dims(), nil), streamctx)
	}
	return errors.New("Unsupported Type")
}
