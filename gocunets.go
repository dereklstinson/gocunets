package gocunets

import (
	"fmt"

	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/layers"
	"github.com/dereklstinson/nccl"
	"gonum.org/v1/gonum/graph"
)

//Tensor is contains 2 tensors the x and dx.  Input IOs will contain only the X tensor.
type Tensor struct {
	*layers.Tensor
	id       int64
	to, from Module
}

//ID implements gonum's graph.Line and Node interfaces
func (t *Tensor) ID() int64 {
	return t.id
}

//From implements gonum's graph.Line interface
func (t *Tensor) From() graph.Node {
	return t.from
}

//To implements gonum's graph.Line interface
func (t *Tensor) To() graph.Node {
	return t.to
}

//ReversedEdge implements graph.Edge interface
func (t *Tensor) ReversedEdge() graph.Edge {
	return &Tensor{
		Tensor: t.Tensor,
		id:     t.id,
		to:     t.from,
		from:   t.to,
	}
}

//ReversedLine implements gonum's graph.Line interface
func (t *Tensor) ReversedLine() graph.Line {
	return &Tensor{
		Tensor: t.Tensor,
		id:     t.id,
		to:     t.from,
		from:   t.to,
	}
}

//Connection is a connection between two operations. It is used for Edge and Line
//This is a test type.  Tensor might be good enough.
type Connection struct {
	y, dy    *layers.Tensor
	id       int64
	to, from Module
}

//ReversedEdge implements graph.Edge interface
func (c *Connection) ReversedEdge() graph.Edge {
	//dy,y might need to be reversed
	return &Connection{
		id:   c.id,
		y:    c.y,
		dy:   c.dy,
		to:   c.from,
		from: c.to,
	}
}

//From implements gonum's graph.Line and graph.Edge interfaces.
func (c *Connection) From() graph.Node {
	return c.from
}

//To implements gonum's graph.Line  and graph.Edge interfaces.
func (c *Connection) To() graph.Node {
	return c.to
}

//ReversedLine implements gonum's graph.Line interface
func (c *Connection) ReversedLine() graph.Line {
	//dy,y might need to be reversed
	return &Connection{
		id:   c.id,
		y:    c.y,
		dy:   c.dy,
		to:   c.from,
		from: c.to,
	}
}

//ID implements gonum's graph.Line and graph.Node interfaces
func (c *Connection) ID() int64 {
	return c.id
}

//CreateWorker assigns a locked host thread to a device.
func CreateWorker(d Device) (w *Worker) {
	w = new(Worker)
	w.Worker = gocu.NewWorker(d.Device)
	return w
}

//Worker is a wrapper for *gocu.Worker  it assigns a locked host thread to a device.
//A device can have more than one worker.
type Worker struct {
	*gocu.Worker
}

//Handle handles the functions of the libraries used in gocunet
type Handle struct {
	*cudnn.Handler
}

//GetWorker returns the gocu.Worker.
func (h *Handle) GetWorker() *Worker {
	return &Worker{
		Worker: h.Handler.Worker,
	}
}

//Comm is a communicator
type Comm struct {
	c   *nccl.Comm
	h   *Handle
	uid nccl.UniqueID
}

//CreateComms creates Communicators for parallel processes.
func CreateComms(hs []*Handle) (comm []*Comm, err error) {
	uid, err := nccl.GetUniqueID()
	if err != nil {
		return nil, err
	}
	nrank := int32(len(hs))
	comm = make([]*Comm, len(hs))
	for i := range hs {
		err = hs[i].Work(func() error {

			comm[i].c, err = nccl.CommInitRank(nrank, uid, int32(i))
			if err != nil {
				return err
			}
			comm[i].h = hs[i]
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return comm, err
}

//Stream is a stream for gpu instructions
type Stream struct {
	*cudart.Stream
}

//SetPeerAccess sets peer access accross all devices
func SetPeerAccess(devs []Device) (connections int, err error) {
	for i := 0; i < len(devs)-1; i++ {
		for j := i + 1; j < len(devs); j++ {
			ok, err := devs[i].CanAccessPeer(devs[j].Device)
			if err != nil {
				return connections, err
			}
			if ok {
				err = devs[i].EnablePeerAccess(devs[j].Device)
				if err != nil {
					return connections, err
				}
				fmt.Println("Connecting i,j", i, j)

				connections++
			} else {

				fmt.Println("Can't Connect i,j", i, j)
			}

		}
	}
	return connections, nil
}

//CreateHandle creates a handle for gocunets
func CreateHandle(w *Worker, d Device, seed uint64) (h *Handle) {
	h = new(Handle)
	h.Handler = cudnn.CreateHandler(w.Worker, d.Device, seed)

	return h
}

//CreateHandles creates parallel handles.  With there own workers.  It also creates non blocking streams
func CreateHandles(ws []*Worker, ds []Device, seeds []uint64) []*Handle {

	hs := make([]*Handle, len(ds))
	var err error
	for i := range ds {
		hs[i] = CreateHandle(ws[i], ds[i], seeds[i])
		err = hs[i].Work(func() error {
			var err error
			stream, err := cudart.CreateNonBlockingStream()
			if err != nil {
				panic(err)
			}
			hs[i].SetStream(stream)
			return nil
		})
		if err != nil {
			panic(err)
		}
	}
	return hs
}

//CreateStream creates a stream
func CreateStream() (s *Stream, err error) {
	s = new(Stream)
	s.Stream, err = cudart.CreateNonBlockingStream()
	return s, err
}

//Device is a gpu device
type Device struct {
	cudart.Device
	num int32
}

//Num is the numerical id of the device
func (d Device) Num() int32 {
	return d.num
}

//GetDeviceList gets a device from a list
func GetDeviceList() (devices []Device, err error) {
	n, err := cudart.GetDeviceCount()
	if err != nil {
		return nil, err
	}
	devices = make([]Device, n)
	for i := (int32)(0); i < n; i++ {
		devices[i].Device = cudart.CreateDevice(i)
		if err != nil {
			return nil, err
		}
		devices[i].num = i
	}
	return devices, nil
}
 