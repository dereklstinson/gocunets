package xactivation

import (
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//UpdateParams updates the params as long as it is set up that way
func (l *Layer) UpdateParams(handle *gocudnn.XHandle, batchsize int) error {
	if l.updateable == false {
		return nil
	}
	l.r.SetBatch(float32(batchsize))

	/*
		l.alphas.T().PrintDeviceMem("alphas: ")
		l.alphas.DeltaT().PrintDeviceMem("dalpha: ")
		l.betas.T().PrintDeviceMem("betas: ")
		l.betas.DeltaT().PrintDeviceMem("dbeta: ")
		l.gsumab.T().PrintDeviceMem("alpha gsum :")
		l.gsumab.DeltaT().PrintDeviceMem("beta gsum :")
		l.xsumab.T().PrintDeviceMem("alpha xsum :")
		l.xsumab.DeltaT().PrintDeviceMem("beta xsum :")
	*/
	err := l.act.UpdateParams(
		handle,
		batchsize,
		l.alphas.T(),
		l.alphas.DeltaT(),
		l.xsumgsum.T(),
		l.xsumgsum.DeltaT(),
		l.l1,
		l.l2,
		l.t,
		l.r)

	if err != nil {
		return err
	}
	if l.memmanaged == true {

		if l.d1 != 0 {
			err = gocudnn.CudaMemCopy(l.l1gptr, l.l1, 4, gocudnn.MemcpyKindFlag{}.Default())
			if err != nil {
				return err
			}
		}
		if l.d2 != 0 {
			err = gocudnn.CudaMemCopy(l.l2gptr, l.l2, 4, gocudnn.MemcpyKindFlag{}.Default())
			if err != nil {
				return err
			}
		}
	} else {
		if l.d1 != 0 {
			err = gocudnn.CudaMemCopy(l.l2gptr, l.l2, 4, gocudnn.MemcpyKindFlag{}.DeviceToHost())
			if err != nil {
				return err
			}
		}
		if l.d2 != 0 {
			err = gocudnn.CudaMemCopy(l.l2gptr, l.l2, 4, gocudnn.MemcpyKindFlag{}.DeviceToHost())
			if err != nil {
				return err
			}
		}

	}
	/*
		l.alphas.T().PrintDeviceMem("alphas: ")
		l.xsumgsum.T().PrintDeviceMem("xsum: ")
		l.xsumgsum.DeltaT().PrintDeviceMem("gsum: ")
		time.Sleep(1000 * time.Millisecond)

	*/
	//fmt.Println("Loss for Parametric L1L2", l.l1g[0], l.l2g[0])
	return nil
}
