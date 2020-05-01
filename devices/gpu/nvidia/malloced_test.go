package nvidia

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/dereklstinson/cutil"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/gocu"
)

func TestMalloced_CreateDevioBuffer(t *testing.T) {
	runtime.LockOSThread()

	check := func(e error) {
		if e != nil {
			t.Error(e)
		}
	}

	dev, err := cudart.GetDevice()
	check(err)
	s, err := cudart.CreateBlockingStream()
	check(err)
	worker := gocu.NewWorker(dev)
	h := cudnn.CreateHandler(worker, dev, 24)
	mem, err := MallocGlobal(h, 256)
	check(err)
	host := make([]byte, 256)
	host2 := make([]byte, 256)
	for i := range host {
		host[i] = (byte)(i)
	}
	hostptr, err := cutil.WrapGoMem(host)
	check(err)
	hostptr2, err := cutil.WrapGoMem(host2)
	check(err)

	check(Memcpy(mem, hostptr, 256))
	check(Memcpy(hostptr2, mem, 256))
	if bytes.Compare(host, host2) != 0 {
		t.Error("Slices not same")
	}
	buf := mem.NewReadWriter(s)

	iobytes, err := ioutil.ReadAll(buf)
	check(err)
	if bytes.Compare(host, iobytes) != 0 {
		fmt.Println(len(host), len(iobytes))
		t.Error("Slices not same", host, iobytes)
	}

}
