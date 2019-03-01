package cudnn

import (
	"github.com/dereklstinson/GoCuNets/devices"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//DataType is used for flags to select what datatype is wanted
type DataType gocudnn.DataType

//Float returns the Float flag
func (d DataType) Float() DataType {
	var dflg gocudnn.DataTypeFlag
	return DataType(dflg.Float())
}

//Double returns the Double flag
func (d DataType) Double() DataType {
	var dflg gocudnn.DataTypeFlag
	return DataType(dflg.Double())
}

//Int32 returns the Int32 flag
func (d DataType) Int32() DataType {
	var dflg gocudnn.DataTypeFlag
	return DataType(dflg.Int32())
}

//UInt8 return the UInt8 flag
func (d DataType) UInt8() DataType {
	var dflg gocudnn.DataTypeFlag
	return DataType(dflg.UInt8())
}

//Int8 returns the Int8 flag
func (d DataType) Int8() DataType {
	var dflg gocudnn.DataTypeFlag
	return DataType(dflg.Int8())
}

//Half returns the Half flag
func (d DataType) Half() DataType {
	var dflg gocudnn.DataTypeFlag
	return DataType(dflg.Half())
}

//Cu returns the gocudnn.DataType Flag.
func (d DataType) Cu() gocudnn.DataType {
	return gocudnn.DataType(d)
}

//Type returns a device.Type.  Will return device.UNKNOWN if not supported
func (d DataType) Type() devices.Type {
	switch d {
	case d.Double():
		return devices.Float64
	case d.Float():
		return devices.Float32
	case d.Half():
		return devices.Float16H
	case d.Int32():
		return devices.Int8
	case d.UInt8():
		return devices.Uint8
	case d.Int8():
		return devices.Int8
	default:
		return devices.UNKNOWN
	}
}
