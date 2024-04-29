package cufft

// This file provides CGO flags to find CUDA libraries and headers.

//#cgo LDFLAGS:-lcufft
//
////default location:
//#cgo LDFLAGS: -L/appl/spack/install-tree/gcc-4.8.5/cuda-10.1.168-v5izax/targets/x86_64-linux/lib
//#cgo CFLAGS: -I/usr/local/cuda/include/ -I/appl/spack/install-tree/gcc-4.8.5/cuda-10.1.168-v5izax/include
//
////Ubuntu 15.04:
//#cgo LDFLAGS:-L/usr/lib/x86_64-linux-gnu/
//#cgo CFLAGS: -I/usr/include
//
////arch linux:
//#cgo LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//#cgo CFLAGS: -I/opt/cuda/include
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/include -w
import "C"
