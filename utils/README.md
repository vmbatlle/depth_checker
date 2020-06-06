# Collection of software utils develop during the project

## example-app (C++)

This is an example of how to use [Monodepth2](https://github.com/nianticlabs/monodepth2)
in a production environment based on C++ (e. g. [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)).

For building, please note the main requirements:

- CUDA Toolkit \[[link](https://developer.nvidia.com/cuda-downloads)\] (we use v11.0)
- cuDNN \[[link](https://developer.nvidia.com/cudnn)\] (we use v7.6.5)
- LibTorch library \[[link](https://pytorch.org/get-started/locally/)\] (we use [v1.5.0 CUDA 10.2](https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.0.zip))
- OpenCV \[[link](https://opencv.org/releases/)\] (we use v3.4.9)

Then do:

```bash
cd example-app
./build.sh
```

For executing, you will need:

- TorchScript files `encoder.cpt` and `encoder.cpt`. They can be generated from modified Monodepth2's Python NN.
  See [corresponding repo](https://github.com/vmbatlle/monodepth2/) for the code. Main modifications were made
  at commit [d1622f1](https://github.com/vmbatlle/monodepth2/commit/d1622f15e4b727f001398d73aad0b503c38122ef).

- A PNG image to use as source for the NN. In the example it's named `0000000.png`.

Then do:

```bash
./build/example-app
```