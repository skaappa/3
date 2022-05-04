package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for regionadds kernel
var regionadds_code cu.Function

// Stores the arguments for regionadds kernel invocation
type regionadds_args_t struct {
	arg_dst     unsafe.Pointer
	arg_LUT     unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_N       int
	argptr      [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for regionadds kernel invocation
var regionadds_args regionadds_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	regionadds_args.argptr[0] = unsafe.Pointer(&regionadds_args.arg_dst)
	regionadds_args.argptr[1] = unsafe.Pointer(&regionadds_args.arg_LUT)
	regionadds_args.argptr[2] = unsafe.Pointer(&regionadds_args.arg_regions)
	regionadds_args.argptr[3] = unsafe.Pointer(&regionadds_args.arg_N)
}

// Wrapper for regionadds CUDA kernel, asynchronous.
func k_regionadds_async(dst unsafe.Pointer, LUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("regionadds")
	}

	regionadds_args.Lock()
	defer regionadds_args.Unlock()

	if regionadds_code == 0 {
		regionadds_code = fatbinLoad(regionadds_map, "regionadds")
	}

	regionadds_args.arg_dst = dst
	regionadds_args.arg_LUT = LUT
	regionadds_args.arg_regions = regions
	regionadds_args.arg_N = N

	args := regionadds_args.argptr[:]
	cu.LaunchKernel(regionadds_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("regionadds")
	}
}

// maps compute capability on PTX code for regionadds kernel.
var regionadds_map = map[int]string{0: "",
	30: regionadds_ptx_30}

// regionadds PTX code for various compute capabilities.
const (
	regionadds_ptx_30 = `
.version 6.4
.target sm_30
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd2;
	ld.global.u8 	%r9, [%rd6];
	mul.wide.u32 	%rd8, %r9, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

BB0_2:
	ret;
}


`
)
