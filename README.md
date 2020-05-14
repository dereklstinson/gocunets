# gocunets


The kame ha me ha of neural networks for Go.  GoCuNets is a GPU centric deep learning framework.  

Pull requests are welcomed.

packages needed

```text
go get github.com/nfnt/resize     // Will eventually get rid of this.
go get github.com/dereklstinson/gpu-monitoring-tools/bindings/go/nvml  //Will eventually get bindings in gocudnn. 
go get github.com/pkg/browser     // This is used with ui it auto launches browser.  Not so useful when used with a headless machine
go get github.com/dereklstinson/nccl
go get github.com/dereklstinson/gocudnn
go get github.com/dereklstinson/cutil
go get -u gonum.org/v1/gonum/...  
go get gonum.org/v1/plot/...      // Will want to get rid of this and use javascript plotting.
 

```

GoCuNets is basically 100% GPU computing using the gocudnn package.  Right now the only gpu support is Nvidia.  Eventually, AMD GPUs will have support through HIP and MIOpen.
I want to make it so that when using this package you don't have to download both cuda and hip libraries.  

This package is separated into a few parts parts
```text
github.com/dereklstinson/gocunets/devices
github.com/dereklstinson/gocunets/layers
github.com/dereklstinson/gocunets/loss
github.com/dereklstinson/gocunets/ui
github.com/dereklstinson/gocunets/trainer
github.com/dereklstinson/gocunets

```

The sub-package devices contains sub-packages for wrappers to device library bindings.  

The sub-package layers contains other sub-packages to layers.  

The sub-package loss contains different loss algorithms.

The sub-package ui contains a makeshift real time neural network monitoring server.  Eventually, I would like to have it create a report / reports.     

The sub-package trainer contains weight trainers.

The main package contains a higher level interface.  

## More on GoCuNets

A lot has changed since the beginning of GoCuNets.  A lot of features that used to be available are not available for the moment.  Like ui interface and model saving and loading.  Saving and loading will be implemented soon.  It will be part an upcoming Model interface.    

type Builder has exposed flags that can be set through their methods.  You only need to set the ones that are going to be used.  A builder builds layers, tensors, randomly initialized tensors and more.  This makes it easier to build networks without having pass flags all the time.  

These flags are 	
```text 
    Frmt TensorFormat
	Dtype     DataType
	Cmode     ConvolutionMode
	Mtype     MathType
	Pmode     PoolingMode
	AMode     ActivationMode
	BNMode    BatchNormMode
	Nan       NanProp
```text

Type Handle is a handle to a GPU.  A GPU can have more than one Handle.  One worker can be used on multiple handles.
Handle and worker must use the same device.

Module interface is new.  Modules will eventually be used to implement graphs.  There are a few modules that I have made.
Eventually everything from layer sub-package will have wrappers in gocunets and will be made into a module.  

Example in how to build a network can be found in ~/go/src/github.com/dereklstinson/gocunets/testing/mnistgputest2

This is a working package, but it is pre alpha.
Better version management will be coming.  

