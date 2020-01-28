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

// CUDA handle for reducedot kernel
var reducedot_code cu.Function

// Stores the arguments for reducedot kernel invocation
type reducedot_args_t struct {
	arg_x1      unsafe.Pointer
	arg_x2      unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducedot kernel invocation
var reducedot_args reducedot_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducedot_args.argptr[0] = unsafe.Pointer(&reducedot_args.arg_x1)
	reducedot_args.argptr[1] = unsafe.Pointer(&reducedot_args.arg_x2)
	reducedot_args.argptr[2] = unsafe.Pointer(&reducedot_args.arg_dst)
	reducedot_args.argptr[3] = unsafe.Pointer(&reducedot_args.arg_initVal)
	reducedot_args.argptr[4] = unsafe.Pointer(&reducedot_args.arg_n)
}

// Wrapper for reducedot CUDA kernel, asynchronous.
func k_reducedot_async(x1 unsafe.Pointer, x2 unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducedot")
	}

	reducedot_args.Lock()
	defer reducedot_args.Unlock()

	if reducedot_code == 0 {
		reducedot_code = fatbinLoad(reducedot_map, "reducedot")
	}

	reducedot_args.arg_x1 = x1
	reducedot_args.arg_x2 = x2
	reducedot_args.arg_dst = dst
	reducedot_args.arg_initVal = initVal
	reducedot_args.arg_n = n

	args := reducedot_args.argptr[:]
	cu.LaunchKernel(reducedot_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducedot")
	}
}

// maps compute capability on PTX code for reducedot kernel.
var reducedot_map = map[int]string{0: "",
	30: reducedot_ptx_30,
	32: reducedot_ptx_32,
	35: reducedot_ptx_35,
	37: reducedot_ptx_37,
	50: reducedot_ptx_50,
	52: reducedot_ptx_52,
	53: reducedot_ptx_53,
	60: reducedot_ptx_60,
	61: reducedot_ptx_61,
	62: reducedot_ptx_62,
	70: reducedot_ptx_70,
	72: reducedot_ptx_72,
	75: reducedot_ptx_75}

// reducedot PTX code for various compute capabilities.
const (
	reducedot_ptx_30 = `
.version 6.5
.target sm_30
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.f32 	%f5, [%rd8];
	ld.global.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_32 = `
.version 6.5
.target sm_32
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_35 = `
.version 6.5
.target sm_35
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_37 = `
.version 6.5
.target sm_37
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_50 = `
.version 6.5
.target sm_50
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_52 = `
.version 6.5
.target sm_52
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_53 = `
.version 6.5
.target sm_53
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_60 = `
.version 6.5
.target sm_60
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_61 = `
.version 6.5
.target sm_61
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_62 = `
.version 6.5
.target sm_62
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_70 = `
.version 6.5
.target sm_70
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_72 = `
.version 6.5
.target sm_72
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
	reducedot_ptx_75 = `
.version 6.5
.target sm_75
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<32>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<10>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd4, [reducedot_param_0];
	ld.param.u64 	%rd5, [reducedot_param_1];
	ld.param.u64 	%rd3, [reducedot_param_2];
	ld.param.f32 	%f31, [reducedot_param_3];
	ld.param.u32 	%r10, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	mov.u32 	%r20, %ntid.x;
	mov.u32 	%r11, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r19, %r20, %r11, %r2;
	mov.u32 	%r12, %nctaid.x;
	mul.lo.s32 	%r4, %r12, %r20;
	setp.ge.s32	%p1, %r19, %r10;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd6, %r19, 4;
	add.s64 	%rd7, %rd2, %rd6;
	add.s64 	%rd8, %rd1, %rd6;
	ld.global.nc.f32 	%f5, [%rd8];
	ld.global.nc.f32 	%f6, [%rd7];
	fma.rn.f32 	%f31, %f6, %f5, %f31;
	add.s32 	%r19, %r19, %r4;
	setp.lt.s32	%p2, %r19, %r10;
	@%p2 bra 	BB0_1;

BB0_2:
	shl.b32 	%r13, %r2, 2;
	mov.u32 	%r14, _ZZ9reducedotE5sdata;
	add.s32 	%r7, %r14, %r13;
	st.shared.f32 	[%r7], %f31;
	bar.sync 	0;
	setp.lt.u32	%p3, %r20, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	shr.u32 	%r9, %r20, 1;
	setp.ge.u32	%p4, %r2, %r9;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f7, [%r7];
	add.s32 	%r15, %r9, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r18, %r14, %r16;
	ld.shared.f32 	%f8, [%r18];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%r7], %f9;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r20, 131;
	mov.u32 	%r20, %r9;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f10, [%r7];
	ld.volatile.shared.f32 	%f11, [%r7+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%r7], %f12;
	ld.volatile.shared.f32 	%f13, [%r7+64];
	ld.volatile.shared.f32 	%f14, [%r7];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%r7], %f15;
	ld.volatile.shared.f32 	%f16, [%r7+32];
	ld.volatile.shared.f32 	%f17, [%r7];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%r7], %f18;
	ld.volatile.shared.f32 	%f19, [%r7+16];
	ld.volatile.shared.f32 	%f20, [%r7];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%r7], %f21;
	ld.volatile.shared.f32 	%f22, [%r7+8];
	ld.volatile.shared.f32 	%f23, [%r7];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%r7], %f24;
	ld.volatile.shared.f32 	%f25, [%r7+4];
	ld.volatile.shared.f32 	%f26, [%r7];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%r7], %f27;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f28, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd9, %rd3;
	atom.global.add.f32 	%f29, [%rd9], %f28;

BB0_10:
	ret;
}


`
)
