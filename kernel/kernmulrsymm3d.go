package kernel

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var kernmulRSymm3D_code cu.Function

type kernmulRSymm3D_args struct {
	arg_fftMx  unsafe.Pointer
	arg_fftMy  unsafe.Pointer
	arg_fftMz  unsafe.Pointer
	arg_fftKxx unsafe.Pointer
	arg_fftKyy unsafe.Pointer
	arg_fftKzz unsafe.Pointer
	arg_fftKyz unsafe.Pointer
	arg_fftKxz unsafe.Pointer
	arg_fftKxy unsafe.Pointer
	arg_N0     int
	arg_N1     int
	arg_N2     int
	argptr     [12]unsafe.Pointer
}

// Wrapper for kernmulRSymm3D CUDA kernel. Synchronizes before return.
func K_kernmulRSymm3D(fftMx unsafe.Pointer, fftMy unsafe.Pointer, fftMz unsafe.Pointer, fftKxx unsafe.Pointer, fftKyy unsafe.Pointer, fftKzz unsafe.Pointer, fftKyz unsafe.Pointer, fftKxz unsafe.Pointer, fftKxy unsafe.Pointer, N0 int, N1 int, N2 int, gridDim, blockDim cu.Dim3) {
	if kernmulRSymm3D_code == 0 {
		kernmulRSymm3D_code = cu.ModuleLoadData(kernmulRSymm3D_ptx).GetFunction("kernmulRSymm3D")
	}

	var a kernmulRSymm3D_args

	a.arg_fftMx = fftMx
	a.argptr[0] = unsafe.Pointer(&a.arg_fftMx)
	a.arg_fftMy = fftMy
	a.argptr[1] = unsafe.Pointer(&a.arg_fftMy)
	a.arg_fftMz = fftMz
	a.argptr[2] = unsafe.Pointer(&a.arg_fftMz)
	a.arg_fftKxx = fftKxx
	a.argptr[3] = unsafe.Pointer(&a.arg_fftKxx)
	a.arg_fftKyy = fftKyy
	a.argptr[4] = unsafe.Pointer(&a.arg_fftKyy)
	a.arg_fftKzz = fftKzz
	a.argptr[5] = unsafe.Pointer(&a.arg_fftKzz)
	a.arg_fftKyz = fftKyz
	a.argptr[6] = unsafe.Pointer(&a.arg_fftKyz)
	a.arg_fftKxz = fftKxz
	a.argptr[7] = unsafe.Pointer(&a.arg_fftKxz)
	a.arg_fftKxy = fftKxy
	a.argptr[8] = unsafe.Pointer(&a.arg_fftKxy)
	a.arg_N0 = N0
	a.argptr[9] = unsafe.Pointer(&a.arg_N0)
	a.arg_N1 = N1
	a.argptr[10] = unsafe.Pointer(&a.arg_N1)
	a.arg_N2 = N2
	a.argptr[11] = unsafe.Pointer(&a.arg_N2)

	args := a.argptr[:]
	str := Stream()
	cu.LaunchKernel(kernmulRSymm3D_code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, 0, str, args)
	SyncAndRecycle(str)
}

const kernmulRSymm3D_ptx = `
.version 3.1
.target sm_30
.address_size 64


.visible .entry kernmulRSymm3D(
	.param .u64 kernmulRSymm3D_param_0,
	.param .u64 kernmulRSymm3D_param_1,
	.param .u64 kernmulRSymm3D_param_2,
	.param .u64 kernmulRSymm3D_param_3,
	.param .u64 kernmulRSymm3D_param_4,
	.param .u64 kernmulRSymm3D_param_5,
	.param .u64 kernmulRSymm3D_param_6,
	.param .u64 kernmulRSymm3D_param_7,
	.param .u64 kernmulRSymm3D_param_8,
	.param .u32 kernmulRSymm3D_param_9,
	.param .u32 kernmulRSymm3D_param_10,
	.param .u32 kernmulRSymm3D_param_11
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<68>;
	.reg .f32 	%f<39>;
	.reg .s64 	%rd<40>;


	ld.param.u64 	%rd3, [kernmulRSymm3D_param_0];
	ld.param.u64 	%rd10, [kernmulRSymm3D_param_1];
	ld.param.u64 	%rd11, [kernmulRSymm3D_param_2];
	ld.param.u64 	%rd4, [kernmulRSymm3D_param_3];
	ld.param.u64 	%rd5, [kernmulRSymm3D_param_4];
	ld.param.u64 	%rd6, [kernmulRSymm3D_param_5];
	ld.param.u64 	%rd7, [kernmulRSymm3D_param_6];
	ld.param.u64 	%rd8, [kernmulRSymm3D_param_7];
	ld.param.u64 	%rd9, [kernmulRSymm3D_param_8];
	ld.param.u32 	%r17, [kernmulRSymm3D_param_9];
	ld.param.u32 	%r18, [kernmulRSymm3D_param_10];
	ld.param.u32 	%r19, [kernmulRSymm3D_param_11];
	cvta.to.global.u64 	%rd1, %rd11;
	cvta.to.global.u64 	%rd2, %rd10;
	.loc 2 35 1
	mov.u32 	%r20, %ntid.y;
	mov.u32 	%r21, %ctaid.y;
	mov.u32 	%r22, %tid.y;
	mad.lo.s32 	%r1, %r20, %r21, %r22;
	.loc 2 36 1
	mov.u32 	%r23, %ntid.x;
	mov.u32 	%r24, %ctaid.x;
	mov.u32 	%r25, %tid.x;
	mad.lo.s32 	%r26, %r23, %r24, %r25;
	.loc 2 38 1
	setp.lt.s32 	%p1, %r26, %r19;
	setp.lt.s32 	%p2, %r1, %r18;
	and.pred  	%p3, %p2, %p1;
	.loc 2 44 1
	setp.gt.s32 	%p4, %r17, 0;
	.loc 2 38 1
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_6;
	bra.uni 	BB0_1;

BB0_1:
	.loc 2 48 1
	shr.u32 	%r28, %r18, 31;
	add.s32 	%r29, %r18, %r28;
	shr.s32 	%r30, %r29, 1;
	add.s32 	%r2, %r30, 1;
	.loc 2 44 1
	sub.s32 	%r39, %r18, %r1;
	mad.lo.s32 	%r67, %r19, %r39, %r26;
	mad.lo.s32 	%r66, %r19, %r1, %r26;
	shl.b32 	%r61, %r66, 1;
	mul.lo.s32 	%r6, %r19, %r18;
	shl.b32 	%r7, %r6, 1;
	mov.u32 	%r62, 0;
	cvta.to.global.u64 	%rd22, %rd8;
	cvta.to.global.u64 	%rd25, %rd6;
	cvta.to.global.u64 	%rd27, %rd5;
	cvta.to.global.u64 	%rd29, %rd4;
	cvta.to.global.u64 	%rd31, %rd3;

BB0_2:
	.loc 2 44 1
	mov.u32 	%r64, %r67;
	mov.u32 	%r63, %r66;
	mov.u32 	%r10, %r64;
	mov.u32 	%r8, %r63;
	setp.lt.s32 	%p6, %r1, %r2;
	.loc 2 48 1
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd12, %rd7;
	.loc 2 60 1
	mul.wide.s32 	%rd13, %r10, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.f32 	%f7, [%rd14];
	neg.f32 	%f38, %f7;
	cvta.to.global.u64 	%rd15, %rd9;
	.loc 2 62 1
	add.s64 	%rd16, %rd15, %rd13;
	ld.global.f32 	%f8, [%rd16];
	neg.f32 	%f37, %f8;
	mov.u32 	%r65, %r10;
	bra.uni 	BB0_5;

BB0_4:
	cvta.to.global.u64 	%rd17, %rd7;
	.loc 2 52 1
	mul.wide.s32 	%rd18, %r8, 4;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f32 	%f38, [%rd19];
	cvta.to.global.u64 	%rd20, %rd9;
	.loc 2 54 1
	add.s64 	%rd21, %rd20, %rd18;
	ld.global.f32 	%f37, [%rd21];
	mov.u32 	%r65, %r8;

BB0_5:
	.loc 2 65 1
	mov.u32 	%r12, %r65;
	.loc 2 53 1
	mul.wide.s32 	%rd23, %r12, 4;
	add.s64 	%rd24, %rd22, %rd23;
	.loc 2 51 1
	add.s64 	%rd26, %rd25, %rd23;
	.loc 2 50 1
	add.s64 	%rd28, %rd27, %rd23;
	.loc 2 49 1
	add.s64 	%rd30, %rd29, %rd23;
	.loc 2 65 1
	ld.global.f32 	%f9, [%rd26];
	ld.global.f32 	%f10, [%rd28];
	.loc 2 66 1
	mul.wide.s32 	%rd32, %r61, 4;
	add.s64 	%rd33, %rd31, %rd32;
	add.s32 	%r46, %r61, 1;
	.loc 2 67 1
	mul.wide.s32 	%rd34, %r46, 4;
	add.s64 	%rd35, %rd31, %rd34;
	ld.global.f32 	%f11, [%rd35];
	.loc 2 68 1
	add.s64 	%rd36, %rd2, %rd32;
	.loc 2 69 1
	add.s64 	%rd37, %rd2, %rd34;
	ld.global.f32 	%f12, [%rd37];
	.loc 2 70 1
	add.s64 	%rd38, %rd1, %rd32;
	.loc 2 71 1
	add.s64 	%rd39, %rd1, %rd34;
	ld.global.f32 	%f13, [%rd39];
	.loc 2 66 1
	ld.global.f32 	%f14, [%rd33];
	.loc 2 65 1
	ld.global.f32 	%f15, [%rd30];
	.loc 2 68 1
	ld.global.f32 	%f16, [%rd36];
	.loc 2 73 1
	mul.f32 	%f17, %f16, %f37;
	fma.rn.f32 	%f18, %f14, %f15, %f17;
	.loc 2 70 1
	ld.global.f32 	%f19, [%rd38];
	.loc 2 65 1
	ld.global.f32 	%f20, [%rd24];
	.loc 2 73 1
	fma.rn.f32 	%f21, %f19, %f20, %f18;
	st.global.f32 	[%rd33], %f21;
	.loc 2 74 1
	mul.f32 	%f22, %f12, %f37;
	fma.rn.f32 	%f23, %f11, %f15, %f22;
	fma.rn.f32 	%f24, %f13, %f20, %f23;
	st.global.f32 	[%rd35], %f24;
	.loc 2 75 1
	mul.f32 	%f25, %f16, %f10;
	fma.rn.f32 	%f26, %f14, %f37, %f25;
	fma.rn.f32 	%f27, %f19, %f38, %f26;
	st.global.f32 	[%rd36], %f27;
	.loc 2 76 1
	mul.f32 	%f28, %f12, %f10;
	fma.rn.f32 	%f29, %f11, %f37, %f28;
	fma.rn.f32 	%f30, %f13, %f38, %f29;
	st.global.f32 	[%rd37], %f30;
	.loc 2 77 1
	mul.f32 	%f31, %f16, %f38;
	fma.rn.f32 	%f32, %f14, %f20, %f31;
	fma.rn.f32 	%f33, %f19, %f9, %f32;
	st.global.f32 	[%rd38], %f33;
	.loc 2 78 1
	mul.f32 	%f34, %f12, %f38;
	fma.rn.f32 	%f35, %f11, %f20, %f34;
	fma.rn.f32 	%f36, %f13, %f9, %f35;
	st.global.f32 	[%rd39], %f36;
	.loc 2 44 1
	add.s32 	%r67, %r10, %r6;
	add.s32 	%r61, %r61, %r7;
	add.s32 	%r66, %r8, %r6;
	.loc 2 44 18
	add.s32 	%r62, %r62, 1;
	.loc 2 44 1
	setp.lt.s32 	%p7, %r62, %r17;
	@%p7 bra 	BB0_2;

BB0_6:
	.loc 2 80 2
	ret;
}


`
