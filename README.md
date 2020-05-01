# gocunets


The kame ha me ha of neural networks in go.

packages needed
'''text
go get github.com/nfnt/resize
go get github.com/dereklstinson/gpu-monitoring-tools/bindings/go/nvml
go get github.com/pkg/browser
go get -u gonum.org/v1/gonum/...
go get gonum.org/v1/plot/...
'''
gocunets is basically 100% cuda computing using the gocudnn package.  So, you will need to go get that. I eventually want to get this to the point were I can straight up build a system through a json file/files.  

Each of the layers has its own package.  I like doing that way to keep things seperated. Because a lot of neural network parts are similar, and I didn't want to have to keep on comming up with elaborate names for structs.  Its just easier to call up the packages to set up the different layers. cnn1:= cnn.LayerSetup(blah blah blah), and then act1:=activation.LayerSetup(bla bla bla). Then you get a cool flow with cnn1.ForwardProp(x,y) act1.ForwardProp(x,y).

I might later add some cpu algos, because you just can't have the cpu sit there while the gpu is doing the heavy lifting.  Especially with something like inference. Where it is not as computationally demanding as when training the networks.  

There is one thing I need to work on is the use of cudnn flags.  Right now you still need to use the gocudnn package to access those flags.  I want to move away from that, and that will happen when I can think of a simple intuitive method of handling those flags. I think I will make it so it there are defaults and can be adjusted through methods.... we will see.



This is a working package, but it is pre alpha. 

