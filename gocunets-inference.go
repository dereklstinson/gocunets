package gocunets

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
)

func (m *Network) buildhiddeniosinference(handle *cudnn.Handler, input *layers.IO) error {
	if len(m.inference.mem) > 0 || m.inference.mem != nil {
		return errors.New("Mem Already Set")
	}
	var previous *layers.IO
	previous = input
	for i := 0; i < len(m.layer)-1; i++ {

		mem, err := m.layer[i].getoutput(handle, previous)
		if err != nil {
			fmt.Println("error in get output")
			return wraperror("getoutputio index: "+strconv.Itoa(i)+" :", err)
		}

		previous = mem
		m.inference.mem = append(m.inference.mem, mem)
	}

	return nil
}

func (m *Network) resizehiddeniosinference(handle *cudnn.Handler, newinput []int32) error {
	var err error
	for i := 0; i < len(m.inference.mem); i++ {
		olddims := m.inference.mem[i].T().Dims()
		newdims := make([]int32, len(olddims))
		copy(newdims, olddims)
		newdims[0] = newinput[0]
		//Since it should only be the batch changing we will just change the batch
		err = m.inference.mem[i].ResizeIO(handle, newdims)
		if err != nil {
			return err
		}

	}
	return nil
}
func (m *Network) Inference(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	var err error
	if m.inference.mem == nil {

		err = m.buildhiddeniosinference(handle, x)
		if err != nil {
			fmt.Println("Error in building hidden os")
			return err
		}
		m.inference.previousdims = x.T().Dims()
		return m.inferenceforward(handle, wspace, x, y)

	}
	_, _, xdims, err := x.Properties()
	if err != nil {
		return err
	}
	if comparedims(m.inference.previousdims, xdims) {
		err = m.inferenceforward(handle, wspace, x, y)
		if err != nil {

			fmt.Println("Error in doing the forward prop after compair dims")

			return err
		}
		return nil
	}

	m.inference.previousdims = xdims
	err = m.resizehiddenios(handle, xdims)
	if err != nil {
		fmt.Println("Error in resize hiddenios")
		return err
	}
	err = m.inferenceforward(handle, wspace, x, y)
	if err != nil {
		fmt.Println("Error in doing the forward prop after resize")
	}
	return nil
}
func (m *Network) inferenceforward(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	var err error

	err = m.layer[0].inference(handle, wspace, x, m.inference.mem[0])
	if err != nil {
		return wraperror("forward index:"+strconv.Itoa(0), err)
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {

		err = m.layer[i].inference(handle, wspace, m.inference.mem[i-1], m.inference.mem[i])
		if err != nil {
			return wraperror("forward index:"+strconv.Itoa(i), err)
		}
	}

	err = m.layer[lnum-1].inference(handle, wspace, m.inference.mem[lnum-2], y)
	if err != nil {

		return wraperror("forward index:"+strconv.Itoa(lnum-1), err)
	}
	return nil
}
