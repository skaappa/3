package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var mul_code cu.Function

type mul_args struct {
	arg_dst unsafe.Pointer
	arg_a   unsafe.Pointer
	arg_b   unsafe.Pointer
	arg_N   int
	argptr  [4]unsafe.Pointer
}

// Wrapper for mul CUDA kernel, asynchronous.
func k_mul_async(dst unsafe.Pointer, a unsafe.Pointer, b unsafe.Pointer, N int, cfg *config, str cu.Stream) {
	if synchronous { // debug
		Sync()
	}

	if mul_code == 0 {
		mul_code = fatbinLoad(mul_map, "mul")
	}

	var _a_ mul_args

	_a_.arg_dst = dst
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_dst)
	_a_.arg_a = a
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_a)
	_a_.arg_b = b
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_b)
	_a_.arg_N = N
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_N)

	args := _a_.argptr[:]
	cu.LaunchKernel(mul_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, str, args)

	if synchronous { // debug
		Sync()
	}
}

// Wrapper for mul CUDA kernel, synchronized.
func k_mul_sync(dst unsafe.Pointer, a unsafe.Pointer, b unsafe.Pointer, N int, cfg *config) {
	Sync()
	k_mul_async(dst, a, b, N, cfg, stream0)
	Sync()
}

var mul_map = map[int]string{0: "",
	20: mul_ptx_20,
	30: mul_ptx_30,
	35: mul_ptx_35}

const (
	mul_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry mul(
	.param .u64 mul_param_0,
	.param .u64 mul_param_1,
	.param .u64 mul_param_2,
	.param .u32 mul_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<4>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd4, [mul_param_0];
	ld.param.u64 	%rd5, [mul_param_1];
	ld.param.u64 	%rd6, [mul_param_2];
	ld.param.u32 	%r2, [mul_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd5;
	.loc 1 5 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 7 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd3, %rd7;
	add.s64 	%rd9, %rd2, %rd7;
	.loc 1 8 1
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd8];
	mul.f32 	%f3, %f2, %f1;
	add.s64 	%rd10, %rd1, %rd7;
	.loc 1 8 1
	st.global.f32 	[%rd10], %f3;

BB0_2:
	.loc 1 10 2
	ret;
}


`
	mul_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry mul(
	.param .u64 mul_param_0,
	.param .u64 mul_param_1,
	.param .u64 mul_param_2,
	.param .u32 mul_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<4>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd4, [mul_param_0];
	ld.param.u64 	%rd5, [mul_param_1];
	ld.param.u64 	%rd6, [mul_param_2];
	ld.param.u32 	%r2, [mul_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd5;
	.loc 1 5 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 7 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd3, %rd7;
	add.s64 	%rd9, %rd2, %rd7;
	.loc 1 8 1
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd8];
	mul.f32 	%f3, %f2, %f1;
	add.s64 	%rd10, %rd1, %rd7;
	.loc 1 8 1
	st.global.f32 	[%rd10], %f3;

BB0_2:
	.loc 1 10 2
	ret;
}


`
	mul_ptx_35 = `
.version 3.2
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 66 3
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 71 3
	ret;
}

.visible .entry mul(
	.param .u64 mul_param_0,
	.param .u64 mul_param_1,
	.param .u64 mul_param_2,
	.param .u32 mul_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<4>;
	.reg .s64 	%rd<11>;


	ld.param.u64 	%rd4, [mul_param_0];
	ld.param.u64 	%rd5, [mul_param_1];
	ld.param.u64 	%rd6, [mul_param_2];
	ld.param.u32 	%r2, [mul_param_3];
	cvta.to.global.u64 	%rd1, %rd4;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd5;
	.loc 1 5 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 7 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB2_2;

	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd3, %rd7;
	add.s64 	%rd9, %rd2, %rd7;
	.loc 1 8 1
	ld.global.nc.f32 	%f1, [%rd9];
	ld.global.nc.f32 	%f2, [%rd8];
	mul.f32 	%f3, %f2, %f1;
	add.s64 	%rd10, %rd1, %rd7;
	.loc 1 8 1
	st.global.f32 	[%rd10], %f3;

BB2_2:
	.loc 1 10 2
	ret;
}


`
)
