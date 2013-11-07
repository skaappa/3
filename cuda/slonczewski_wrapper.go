package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var addslonczewskitorque_code cu.Function

type addslonczewskitorque_args struct {
	arg_tx              unsafe.Pointer
	arg_ty              unsafe.Pointer
	arg_tz              unsafe.Pointer
	arg_mx              unsafe.Pointer
	arg_my              unsafe.Pointer
	arg_mz              unsafe.Pointer
	arg_jz              unsafe.Pointer
	arg_pxLUT           unsafe.Pointer
	arg_pyLUT           unsafe.Pointer
	arg_pzLUT           unsafe.Pointer
	arg_msatLUT         unsafe.Pointer
	arg_alphaLUT        unsafe.Pointer
	arg_flt             float32
	arg_polLUT          unsafe.Pointer
	arg_lambdaLUT       unsafe.Pointer
	arg_epsilonPrimeLUT unsafe.Pointer
	arg_regions         unsafe.Pointer
	arg_N               int
	argptr              [18]unsafe.Pointer
}

// Wrapper for addslonczewskitorque CUDA kernel, asynchronous.
func k_addslonczewskitorque_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, jz unsafe.Pointer, pxLUT unsafe.Pointer, pyLUT unsafe.Pointer, pzLUT unsafe.Pointer, msatLUT unsafe.Pointer, alphaLUT unsafe.Pointer, flt float32, polLUT unsafe.Pointer, lambdaLUT unsafe.Pointer, epsilonPrimeLUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config, str cu.Stream) {
	if synchronous { // debug
		Sync()
	}

	if addslonczewskitorque_code == 0 {
		addslonczewskitorque_code = fatbinLoad(addslonczewskitorque_map, "addslonczewskitorque")
	}

	var _a_ addslonczewskitorque_args

	_a_.arg_tx = tx
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_tx)
	_a_.arg_ty = ty
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_ty)
	_a_.arg_tz = tz
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_tz)
	_a_.arg_mx = mx
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_mx)
	_a_.arg_my = my
	_a_.argptr[4] = unsafe.Pointer(&_a_.arg_my)
	_a_.arg_mz = mz
	_a_.argptr[5] = unsafe.Pointer(&_a_.arg_mz)
	_a_.arg_jz = jz
	_a_.argptr[6] = unsafe.Pointer(&_a_.arg_jz)
	_a_.arg_pxLUT = pxLUT
	_a_.argptr[7] = unsafe.Pointer(&_a_.arg_pxLUT)
	_a_.arg_pyLUT = pyLUT
	_a_.argptr[8] = unsafe.Pointer(&_a_.arg_pyLUT)
	_a_.arg_pzLUT = pzLUT
	_a_.argptr[9] = unsafe.Pointer(&_a_.arg_pzLUT)
	_a_.arg_msatLUT = msatLUT
	_a_.argptr[10] = unsafe.Pointer(&_a_.arg_msatLUT)
	_a_.arg_alphaLUT = alphaLUT
	_a_.argptr[11] = unsafe.Pointer(&_a_.arg_alphaLUT)
	_a_.arg_flt = flt
	_a_.argptr[12] = unsafe.Pointer(&_a_.arg_flt)
	_a_.arg_polLUT = polLUT
	_a_.argptr[13] = unsafe.Pointer(&_a_.arg_polLUT)
	_a_.arg_lambdaLUT = lambdaLUT
	_a_.argptr[14] = unsafe.Pointer(&_a_.arg_lambdaLUT)
	_a_.arg_epsilonPrimeLUT = epsilonPrimeLUT
	_a_.argptr[15] = unsafe.Pointer(&_a_.arg_epsilonPrimeLUT)
	_a_.arg_regions = regions
	_a_.argptr[16] = unsafe.Pointer(&_a_.arg_regions)
	_a_.arg_N = N
	_a_.argptr[17] = unsafe.Pointer(&_a_.arg_N)

	args := _a_.argptr[:]
	cu.LaunchKernel(addslonczewskitorque_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, str, args)

	if synchronous { // debug
		Sync()
	}
}

// Wrapper for addslonczewskitorque CUDA kernel, synchronized.
func k_addslonczewskitorque_sync(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, jz unsafe.Pointer, pxLUT unsafe.Pointer, pyLUT unsafe.Pointer, pzLUT unsafe.Pointer, msatLUT unsafe.Pointer, alphaLUT unsafe.Pointer, flt float32, polLUT unsafe.Pointer, lambdaLUT unsafe.Pointer, epsilonPrimeLUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	Sync()
	k_addslonczewskitorque_async(tx, ty, tz, mx, my, mz, jz, pxLUT, pyLUT, pzLUT, msatLUT, alphaLUT, flt, polLUT, lambdaLUT, epsilonPrimeLUT, regions, N, cfg, stream0)
	Sync()
}

var addslonczewskitorque_map = map[int]string{0: "",
	20: addslonczewskitorque_ptx_20,
	30: addslonczewskitorque_ptx_30,
	35: addslonczewskitorque_ptx_35}

const (
	addslonczewskitorque_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry addslonczewskitorque(
	.param .u64 addslonczewskitorque_param_0,
	.param .u64 addslonczewskitorque_param_1,
	.param .u64 addslonczewskitorque_param_2,
	.param .u64 addslonczewskitorque_param_3,
	.param .u64 addslonczewskitorque_param_4,
	.param .u64 addslonczewskitorque_param_5,
	.param .u64 addslonczewskitorque_param_6,
	.param .u64 addslonczewskitorque_param_7,
	.param .u64 addslonczewskitorque_param_8,
	.param .u64 addslonczewskitorque_param_9,
	.param .u64 addslonczewskitorque_param_10,
	.param .u64 addslonczewskitorque_param_11,
	.param .f32 addslonczewskitorque_param_12,
	.param .u64 addslonczewskitorque_param_13,
	.param .u64 addslonczewskitorque_param_14,
	.param .u64 addslonczewskitorque_param_15,
	.param .u64 addslonczewskitorque_param_16,
	.param .u32 addslonczewskitorque_param_17
)
{
	.reg .pred 	%p<6>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<18>;
	.reg .f32 	%f<77>;
	.reg .s64 	%rd<56>;
	.reg .f64 	%fd<3>;


	ld.param.u64 	%rd2, [addslonczewskitorque_param_0];
	ld.param.u64 	%rd3, [addslonczewskitorque_param_1];
	ld.param.u64 	%rd4, [addslonczewskitorque_param_2];
	ld.param.u64 	%rd5, [addslonczewskitorque_param_3];
	ld.param.u64 	%rd6, [addslonczewskitorque_param_4];
	ld.param.u64 	%rd7, [addslonczewskitorque_param_5];
	ld.param.u64 	%rd8, [addslonczewskitorque_param_6];
	ld.param.u64 	%rd9, [addslonczewskitorque_param_7];
	ld.param.u64 	%rd10, [addslonczewskitorque_param_8];
	ld.param.u64 	%rd11, [addslonczewskitorque_param_9];
	ld.param.u64 	%rd12, [addslonczewskitorque_param_10];
	ld.param.u64 	%rd13, [addslonczewskitorque_param_11];
	ld.param.f32 	%f19, [addslonczewskitorque_param_12];
	ld.param.u64 	%rd14, [addslonczewskitorque_param_13];
	ld.param.u64 	%rd15, [addslonczewskitorque_param_14];
	ld.param.u64 	%rd16, [addslonczewskitorque_param_15];
	ld.param.u64 	%rd17, [addslonczewskitorque_param_16];
	ld.param.u32 	%r3, [addslonczewskitorque_param_17];
	cvta.to.global.u64 	%rd1, %rd17;
	.loc 1 15 1
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	.loc 1 16 1
	setp.ge.s32	%p1, %r1, %r3;
	@%p1 bra 	BB0_6;

	cvta.to.global.u64 	%rd18, %rd5;
	mul.wide.s32 	%rd19, %r1, 4;
	add.s64 	%rd20, %rd18, %rd19;
	cvt.s64.s32	%rd21, %r1;
	.loc 1 18 1
	ld.global.f32 	%f1, [%rd20];
	cvta.to.global.u64 	%rd22, %rd6;
	add.s64 	%rd23, %rd22, %rd19;
	.loc 1 18 1
	ld.global.f32 	%f2, [%rd23];
	cvta.to.global.u64 	%rd24, %rd7;
	add.s64 	%rd25, %rd24, %rd19;
	.loc 1 18 1
	ld.global.f32 	%f3, [%rd25];
	cvta.to.global.u64 	%rd26, %rd8;
	add.s64 	%rd27, %rd26, %rd19;
	.loc 1 19 1
	ld.global.f32 	%f4, [%rd27];
	add.s64 	%rd28, %rd1, %rd21;
	.loc 1 22 1
	ld.global.u8 	%rs1, [%rd28];
	cvt.u32.u16	%r10, %rs1;
	cvt.s32.s8 	%r2, %r10;
	cvta.to.global.u64 	%rd29, %rd9;
	cvt.u64.u16	%rd30, %rs1;
	cvt.s64.s8 	%rd31, %rd30;
	shl.b64 	%rd32, %rd31, 2;
	add.s64 	%rd33, %rd29, %rd32;
	cvta.to.global.u64 	%rd34, %rd10;
	add.s64 	%rd35, %rd34, %rd32;
	cvta.to.global.u64 	%rd36, %rd11;
	add.s64 	%rd37, %rd36, %rd32;
	.loc 1 24 1
	ld.global.f32 	%f5, [%rd33];
	ld.global.f32 	%f6, [%rd35];
	mul.f32 	%f20, %f6, %f6;
	fma.rn.f32 	%f21, %f5, %f5, %f20;
	ld.global.f32 	%f7, [%rd37];
	fma.rn.f32 	%f22, %f7, %f7, %f21;
	.loc 2 3055 10
	sqrt.rn.f32 	%f8, %f22;
	setp.neu.f32	%p2, %f8, 0f00000000;
	@%p2 bra 	BB0_3;

	mov.f32 	%f76, 0f00000000;
	bra.uni 	BB0_4;

BB0_3:
	rcp.rn.f32 	%f76, %f8;

BB0_4:
	mul.f32 	%f11, %f76, %f5;
	mul.f32 	%f12, %f76, %f6;
	mul.f32 	%f13, %f76, %f7;
	cvta.to.global.u64 	%rd38, %rd12;
	mul.wide.s32 	%rd39, %r2, 4;
	add.s64 	%rd40, %rd38, %rd39;
	cvta.to.global.u64 	%rd41, %rd13;
	add.s64 	%rd42, %rd41, %rd39;
	.loc 1 26 1
	ld.global.f32 	%f14, [%rd42];
	cvta.to.global.u64 	%rd43, %rd14;
	add.s64 	%rd44, %rd43, %rd39;
	.loc 1 27 1
	ld.global.f32 	%f15, [%rd44];
	cvta.to.global.u64 	%rd45, %rd15;
	add.s64 	%rd46, %rd45, %rd39;
	.loc 1 28 1
	ld.global.f32 	%f16, [%rd46];
	cvta.to.global.u64 	%rd47, %rd16;
	add.s64 	%rd48, %rd47, %rd39;
	.loc 1 29 1
	ld.global.f32 	%f17, [%rd48];
	.loc 1 25 1
	ld.global.f32 	%f18, [%rd40];
	.loc 1 31 1
	setp.eq.f32	%p3, %f18, 0f00000000;
	setp.eq.f32	%p4, %f4, 0f00000000;
	or.pred  	%p5, %p4, %p3;
	.loc 1 31 1
	@%p5 bra 	BB0_6;

	.loc 1 35 1
	mul.f32 	%f24, %f18, %f19;
	.loc 2 3608 3
	div.rn.f32 	%f25, %f4, %f24;
	.loc 1 35 92
	cvt.f64.f32	%fd1, %f25;
	mul.f64 	%fd2, %fd1, 0d3CC7B6EF14E9250C;
	cvt.rn.f32.f64	%f26, %fd2;
	.loc 1 36 1
	mul.f32 	%f27, %f16, %f16;
	.loc 1 37 1
	mul.f32 	%f28, %f15, %f27;
	add.f32 	%f29, %f27, 0f3F800000;
	add.f32 	%f30, %f27, 0fBF800000;
	mul.f32 	%f31, %f12, %f2;
	fma.rn.f32 	%f32, %f11, %f1, %f31;
	fma.rn.f32 	%f33, %f13, %f3, %f32;
	fma.rn.f32 	%f34, %f30, %f33, %f29;
	.loc 2 3608 3
	div.rn.f32 	%f35, %f28, %f34;
	.loc 1 39 1
	mul.f32 	%f36, %f26, %f35;
	.loc 1 40 1
	mul.f32 	%f37, %f26, %f17;
	.loc 1 42 1
	fma.rn.f32 	%f38, %f14, %f14, 0f3F800000;
	rcp.rn.f32 	%f39, %f38;
	.loc 1 43 1
	mul.f32 	%f40, %f14, %f37;
	sub.f32 	%f41, %f36, %f40;
	mul.f32 	%f42, %f39, %f41;
	.loc 1 44 1
	mul.f32 	%f43, %f14, %f36;
	sub.f32 	%f44, %f37, %f43;
	mul.f32 	%f45, %f39, %f44;
	.loc 1 46 1
	mul.f32 	%f46, %f13, %f2;
	mul.f32 	%f47, %f12, %f3;
	sub.f32 	%f48, %f47, %f46;
	mul.f32 	%f49, %f11, %f3;
	mul.f32 	%f50, %f13, %f1;
	sub.f32 	%f51, %f50, %f49;
	mul.f32 	%f52, %f12, %f1;
	mul.f32 	%f53, %f11, %f2;
	sub.f32 	%f54, %f53, %f52;
	.loc 1 47 1
	mul.f32 	%f55, %f2, %f54;
	mul.f32 	%f56, %f3, %f51;
	sub.f32 	%f57, %f55, %f56;
	mul.f32 	%f58, %f3, %f48;
	mul.f32 	%f59, %f1, %f54;
	sub.f32 	%f60, %f58, %f59;
	mul.f32 	%f61, %f1, %f51;
	mul.f32 	%f62, %f2, %f48;
	sub.f32 	%f63, %f61, %f62;
	.loc 1 49 1
	mul.f32 	%f64, %f45, %f48;
	fma.rn.f32 	%f65, %f42, %f57, %f64;
	cvta.to.global.u64 	%rd49, %rd2;
	mul.wide.s32 	%rd50, %r1, 4;
	add.s64 	%rd51, %rd49, %rd50;
	.loc 1 49 1
	ld.global.f32 	%f66, [%rd51];
	add.f32 	%f67, %f66, %f65;
	st.global.f32 	[%rd51], %f67;
	.loc 1 50 1
	mul.f32 	%f68, %f45, %f51;
	fma.rn.f32 	%f69, %f42, %f60, %f68;
	cvta.to.global.u64 	%rd52, %rd3;
	add.s64 	%rd53, %rd52, %rd50;
	.loc 1 50 1
	ld.global.f32 	%f70, [%rd53];
	add.f32 	%f71, %f70, %f69;
	st.global.f32 	[%rd53], %f71;
	.loc 1 51 1
	mul.f32 	%f72, %f45, %f54;
	fma.rn.f32 	%f73, %f42, %f63, %f72;
	cvta.to.global.u64 	%rd54, %rd4;
	add.s64 	%rd55, %rd54, %rd50;
	.loc 1 51 1
	ld.global.f32 	%f74, [%rd55];
	add.f32 	%f75, %f74, %f73;
	st.global.f32 	[%rd55], %f75;

BB0_6:
	.loc 1 53 2
	ret;
}


`
	addslonczewskitorque_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry addslonczewskitorque(
	.param .u64 addslonczewskitorque_param_0,
	.param .u64 addslonczewskitorque_param_1,
	.param .u64 addslonczewskitorque_param_2,
	.param .u64 addslonczewskitorque_param_3,
	.param .u64 addslonczewskitorque_param_4,
	.param .u64 addslonczewskitorque_param_5,
	.param .u64 addslonczewskitorque_param_6,
	.param .u64 addslonczewskitorque_param_7,
	.param .u64 addslonczewskitorque_param_8,
	.param .u64 addslonczewskitorque_param_9,
	.param .u64 addslonczewskitorque_param_10,
	.param .u64 addslonczewskitorque_param_11,
	.param .f32 addslonczewskitorque_param_12,
	.param .u64 addslonczewskitorque_param_13,
	.param .u64 addslonczewskitorque_param_14,
	.param .u64 addslonczewskitorque_param_15,
	.param .u64 addslonczewskitorque_param_16,
	.param .u32 addslonczewskitorque_param_17
)
{
	.reg .pred 	%p<6>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<18>;
	.reg .f32 	%f<77>;
	.reg .s64 	%rd<56>;
	.reg .f64 	%fd<3>;


	ld.param.u64 	%rd2, [addslonczewskitorque_param_0];
	ld.param.u64 	%rd3, [addslonczewskitorque_param_1];
	ld.param.u64 	%rd4, [addslonczewskitorque_param_2];
	ld.param.u64 	%rd5, [addslonczewskitorque_param_3];
	ld.param.u64 	%rd6, [addslonczewskitorque_param_4];
	ld.param.u64 	%rd7, [addslonczewskitorque_param_5];
	ld.param.u64 	%rd8, [addslonczewskitorque_param_6];
	ld.param.u64 	%rd9, [addslonczewskitorque_param_7];
	ld.param.u64 	%rd10, [addslonczewskitorque_param_8];
	ld.param.u64 	%rd11, [addslonczewskitorque_param_9];
	ld.param.u64 	%rd12, [addslonczewskitorque_param_10];
	ld.param.u64 	%rd13, [addslonczewskitorque_param_11];
	ld.param.f32 	%f19, [addslonczewskitorque_param_12];
	ld.param.u64 	%rd14, [addslonczewskitorque_param_13];
	ld.param.u64 	%rd15, [addslonczewskitorque_param_14];
	ld.param.u64 	%rd16, [addslonczewskitorque_param_15];
	ld.param.u64 	%rd17, [addslonczewskitorque_param_16];
	ld.param.u32 	%r3, [addslonczewskitorque_param_17];
	cvta.to.global.u64 	%rd1, %rd17;
	.loc 1 15 1
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	.loc 1 16 1
	setp.ge.s32	%p1, %r1, %r3;
	@%p1 bra 	BB0_6;

	cvta.to.global.u64 	%rd18, %rd5;
	mul.wide.s32 	%rd19, %r1, 4;
	add.s64 	%rd20, %rd18, %rd19;
	cvt.s64.s32	%rd21, %r1;
	.loc 1 18 1
	ld.global.f32 	%f1, [%rd20];
	cvta.to.global.u64 	%rd22, %rd6;
	add.s64 	%rd23, %rd22, %rd19;
	.loc 1 18 1
	ld.global.f32 	%f2, [%rd23];
	cvta.to.global.u64 	%rd24, %rd7;
	add.s64 	%rd25, %rd24, %rd19;
	.loc 1 18 1
	ld.global.f32 	%f3, [%rd25];
	cvta.to.global.u64 	%rd26, %rd8;
	add.s64 	%rd27, %rd26, %rd19;
	.loc 1 19 1
	ld.global.f32 	%f4, [%rd27];
	add.s64 	%rd28, %rd1, %rd21;
	.loc 1 22 1
	ld.global.u8 	%rs1, [%rd28];
	cvt.u32.u16	%r10, %rs1;
	cvt.s32.s8 	%r2, %r10;
	cvta.to.global.u64 	%rd29, %rd9;
	cvt.u64.u16	%rd30, %rs1;
	cvt.s64.s8 	%rd31, %rd30;
	shl.b64 	%rd32, %rd31, 2;
	add.s64 	%rd33, %rd29, %rd32;
	cvta.to.global.u64 	%rd34, %rd10;
	add.s64 	%rd35, %rd34, %rd32;
	cvta.to.global.u64 	%rd36, %rd11;
	add.s64 	%rd37, %rd36, %rd32;
	.loc 1 24 1
	ld.global.f32 	%f5, [%rd33];
	ld.global.f32 	%f6, [%rd35];
	mul.f32 	%f20, %f6, %f6;
	fma.rn.f32 	%f21, %f5, %f5, %f20;
	ld.global.f32 	%f7, [%rd37];
	fma.rn.f32 	%f22, %f7, %f7, %f21;
	.loc 2 3055 10
	sqrt.rn.f32 	%f8, %f22;
	setp.neu.f32	%p2, %f8, 0f00000000;
	@%p2 bra 	BB0_3;

	mov.f32 	%f76, 0f00000000;
	bra.uni 	BB0_4;

BB0_3:
	rcp.rn.f32 	%f76, %f8;

BB0_4:
	mul.f32 	%f11, %f76, %f5;
	mul.f32 	%f12, %f76, %f6;
	mul.f32 	%f13, %f76, %f7;
	cvta.to.global.u64 	%rd38, %rd12;
	mul.wide.s32 	%rd39, %r2, 4;
	add.s64 	%rd40, %rd38, %rd39;
	cvta.to.global.u64 	%rd41, %rd13;
	add.s64 	%rd42, %rd41, %rd39;
	.loc 1 26 1
	ld.global.f32 	%f14, [%rd42];
	cvta.to.global.u64 	%rd43, %rd14;
	add.s64 	%rd44, %rd43, %rd39;
	.loc 1 27 1
	ld.global.f32 	%f15, [%rd44];
	cvta.to.global.u64 	%rd45, %rd15;
	add.s64 	%rd46, %rd45, %rd39;
	.loc 1 28 1
	ld.global.f32 	%f16, [%rd46];
	cvta.to.global.u64 	%rd47, %rd16;
	add.s64 	%rd48, %rd47, %rd39;
	.loc 1 29 1
	ld.global.f32 	%f17, [%rd48];
	.loc 1 25 1
	ld.global.f32 	%f18, [%rd40];
	.loc 1 31 1
	setp.eq.f32	%p3, %f18, 0f00000000;
	setp.eq.f32	%p4, %f4, 0f00000000;
	or.pred  	%p5, %p4, %p3;
	.loc 1 31 1
	@%p5 bra 	BB0_6;

	.loc 1 35 1
	mul.f32 	%f24, %f18, %f19;
	.loc 2 3608 3
	div.rn.f32 	%f25, %f4, %f24;
	.loc 1 35 92
	cvt.f64.f32	%fd1, %f25;
	mul.f64 	%fd2, %fd1, 0d3CC7B6EF14E9250C;
	cvt.rn.f32.f64	%f26, %fd2;
	.loc 1 36 1
	mul.f32 	%f27, %f16, %f16;
	.loc 1 37 1
	mul.f32 	%f28, %f15, %f27;
	add.f32 	%f29, %f27, 0f3F800000;
	add.f32 	%f30, %f27, 0fBF800000;
	mul.f32 	%f31, %f12, %f2;
	fma.rn.f32 	%f32, %f11, %f1, %f31;
	fma.rn.f32 	%f33, %f13, %f3, %f32;
	fma.rn.f32 	%f34, %f30, %f33, %f29;
	.loc 2 3608 3
	div.rn.f32 	%f35, %f28, %f34;
	.loc 1 39 1
	mul.f32 	%f36, %f26, %f35;
	.loc 1 40 1
	mul.f32 	%f37, %f26, %f17;
	.loc 1 42 1
	fma.rn.f32 	%f38, %f14, %f14, 0f3F800000;
	rcp.rn.f32 	%f39, %f38;
	.loc 1 43 1
	mul.f32 	%f40, %f14, %f37;
	sub.f32 	%f41, %f36, %f40;
	mul.f32 	%f42, %f39, %f41;
	.loc 1 44 1
	mul.f32 	%f43, %f14, %f36;
	sub.f32 	%f44, %f37, %f43;
	mul.f32 	%f45, %f39, %f44;
	.loc 1 46 1
	mul.f32 	%f46, %f13, %f2;
	mul.f32 	%f47, %f12, %f3;
	sub.f32 	%f48, %f47, %f46;
	mul.f32 	%f49, %f11, %f3;
	mul.f32 	%f50, %f13, %f1;
	sub.f32 	%f51, %f50, %f49;
	mul.f32 	%f52, %f12, %f1;
	mul.f32 	%f53, %f11, %f2;
	sub.f32 	%f54, %f53, %f52;
	.loc 1 47 1
	mul.f32 	%f55, %f2, %f54;
	mul.f32 	%f56, %f3, %f51;
	sub.f32 	%f57, %f55, %f56;
	mul.f32 	%f58, %f3, %f48;
	mul.f32 	%f59, %f1, %f54;
	sub.f32 	%f60, %f58, %f59;
	mul.f32 	%f61, %f1, %f51;
	mul.f32 	%f62, %f2, %f48;
	sub.f32 	%f63, %f61, %f62;
	.loc 1 49 1
	mul.f32 	%f64, %f45, %f48;
	fma.rn.f32 	%f65, %f42, %f57, %f64;
	cvta.to.global.u64 	%rd49, %rd2;
	mul.wide.s32 	%rd50, %r1, 4;
	add.s64 	%rd51, %rd49, %rd50;
	.loc 1 49 1
	ld.global.f32 	%f66, [%rd51];
	add.f32 	%f67, %f66, %f65;
	st.global.f32 	[%rd51], %f67;
	.loc 1 50 1
	mul.f32 	%f68, %f45, %f51;
	fma.rn.f32 	%f69, %f42, %f60, %f68;
	cvta.to.global.u64 	%rd52, %rd3;
	add.s64 	%rd53, %rd52, %rd50;
	.loc 1 50 1
	ld.global.f32 	%f70, [%rd53];
	add.f32 	%f71, %f70, %f69;
	st.global.f32 	[%rd53], %f71;
	.loc 1 51 1
	mul.f32 	%f72, %f45, %f54;
	fma.rn.f32 	%f73, %f42, %f63, %f72;
	cvta.to.global.u64 	%rd54, %rd4;
	add.s64 	%rd55, %rd54, %rd50;
	.loc 1 51 1
	ld.global.f32 	%f74, [%rd55];
	add.f32 	%f75, %f74, %f73;
	st.global.f32 	[%rd55], %f75;

BB0_6:
	.loc 1 53 2
	ret;
}


`
	addslonczewskitorque_ptx_35 = `
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

.visible .entry addslonczewskitorque(
	.param .u64 addslonczewskitorque_param_0,
	.param .u64 addslonczewskitorque_param_1,
	.param .u64 addslonczewskitorque_param_2,
	.param .u64 addslonczewskitorque_param_3,
	.param .u64 addslonczewskitorque_param_4,
	.param .u64 addslonczewskitorque_param_5,
	.param .u64 addslonczewskitorque_param_6,
	.param .u64 addslonczewskitorque_param_7,
	.param .u64 addslonczewskitorque_param_8,
	.param .u64 addslonczewskitorque_param_9,
	.param .u64 addslonczewskitorque_param_10,
	.param .u64 addslonczewskitorque_param_11,
	.param .f32 addslonczewskitorque_param_12,
	.param .u64 addslonczewskitorque_param_13,
	.param .u64 addslonczewskitorque_param_14,
	.param .u64 addslonczewskitorque_param_15,
	.param .u64 addslonczewskitorque_param_16,
	.param .u32 addslonczewskitorque_param_17
)
{
	.reg .pred 	%p<6>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<18>;
	.reg .f32 	%f<77>;
	.reg .s64 	%rd<56>;
	.reg .f64 	%fd<3>;


	ld.param.u64 	%rd2, [addslonczewskitorque_param_0];
	ld.param.u64 	%rd3, [addslonczewskitorque_param_1];
	ld.param.u64 	%rd4, [addslonczewskitorque_param_2];
	ld.param.u64 	%rd5, [addslonczewskitorque_param_3];
	ld.param.u64 	%rd6, [addslonczewskitorque_param_4];
	ld.param.u64 	%rd7, [addslonczewskitorque_param_5];
	ld.param.u64 	%rd8, [addslonczewskitorque_param_6];
	ld.param.u64 	%rd9, [addslonczewskitorque_param_7];
	ld.param.u64 	%rd10, [addslonczewskitorque_param_8];
	ld.param.u64 	%rd11, [addslonczewskitorque_param_9];
	ld.param.u64 	%rd12, [addslonczewskitorque_param_10];
	ld.param.u64 	%rd13, [addslonczewskitorque_param_11];
	ld.param.f32 	%f19, [addslonczewskitorque_param_12];
	ld.param.u64 	%rd14, [addslonczewskitorque_param_13];
	ld.param.u64 	%rd15, [addslonczewskitorque_param_14];
	ld.param.u64 	%rd16, [addslonczewskitorque_param_15];
	ld.param.u64 	%rd17, [addslonczewskitorque_param_16];
	ld.param.u32 	%r3, [addslonczewskitorque_param_17];
	cvta.to.global.u64 	%rd1, %rd17;
	.loc 1 15 1
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	.loc 1 16 1
	setp.ge.s32	%p1, %r1, %r3;
	@%p1 bra 	BB2_6;

	cvta.to.global.u64 	%rd18, %rd5;
	mul.wide.s32 	%rd19, %r1, 4;
	add.s64 	%rd20, %rd18, %rd19;
	cvt.s64.s32	%rd21, %r1;
	.loc 1 18 1
	ld.global.nc.f32 	%f1, [%rd20];
	cvta.to.global.u64 	%rd22, %rd6;
	add.s64 	%rd23, %rd22, %rd19;
	.loc 1 18 1
	ld.global.nc.f32 	%f2, [%rd23];
	cvta.to.global.u64 	%rd24, %rd7;
	add.s64 	%rd25, %rd24, %rd19;
	.loc 1 18 1
	ld.global.nc.f32 	%f3, [%rd25];
	cvta.to.global.u64 	%rd26, %rd8;
	add.s64 	%rd27, %rd26, %rd19;
	.loc 1 19 1
	ld.global.nc.f32 	%f4, [%rd27];
	add.s64 	%rd28, %rd1, %rd21;
	.loc 1 22 1
	ld.global.nc.u8 	%rs1, [%rd28];
	cvt.u32.u16	%r10, %rs1;
	cvt.s32.s8 	%r2, %r10;
	cvta.to.global.u64 	%rd29, %rd9;
	cvt.u64.u16	%rd30, %rs1;
	cvt.s64.s8 	%rd31, %rd30;
	shl.b64 	%rd32, %rd31, 2;
	add.s64 	%rd33, %rd29, %rd32;
	cvta.to.global.u64 	%rd34, %rd10;
	add.s64 	%rd35, %rd34, %rd32;
	cvta.to.global.u64 	%rd36, %rd11;
	add.s64 	%rd37, %rd36, %rd32;
	.loc 1 24 1
	ld.global.nc.f32 	%f5, [%rd33];
	ld.global.nc.f32 	%f6, [%rd35];
	mul.f32 	%f20, %f6, %f6;
	fma.rn.f32 	%f21, %f5, %f5, %f20;
	ld.global.nc.f32 	%f7, [%rd37];
	fma.rn.f32 	%f22, %f7, %f7, %f21;
	.loc 3 3055 10
	sqrt.rn.f32 	%f8, %f22;
	setp.neu.f32	%p2, %f8, 0f00000000;
	@%p2 bra 	BB2_3;

	mov.f32 	%f76, 0f00000000;
	bra.uni 	BB2_4;

BB2_3:
	rcp.rn.f32 	%f76, %f8;

BB2_4:
	mul.f32 	%f11, %f76, %f5;
	mul.f32 	%f12, %f76, %f6;
	mul.f32 	%f13, %f76, %f7;
	cvta.to.global.u64 	%rd38, %rd12;
	mul.wide.s32 	%rd39, %r2, 4;
	add.s64 	%rd40, %rd38, %rd39;
	cvta.to.global.u64 	%rd41, %rd13;
	add.s64 	%rd42, %rd41, %rd39;
	.loc 1 26 1
	ld.global.nc.f32 	%f14, [%rd42];
	cvta.to.global.u64 	%rd43, %rd14;
	add.s64 	%rd44, %rd43, %rd39;
	.loc 1 27 1
	ld.global.nc.f32 	%f15, [%rd44];
	cvta.to.global.u64 	%rd45, %rd15;
	add.s64 	%rd46, %rd45, %rd39;
	.loc 1 28 1
	ld.global.nc.f32 	%f16, [%rd46];
	cvta.to.global.u64 	%rd47, %rd16;
	add.s64 	%rd48, %rd47, %rd39;
	.loc 1 29 1
	ld.global.nc.f32 	%f17, [%rd48];
	.loc 1 25 1
	ld.global.nc.f32 	%f18, [%rd40];
	.loc 1 31 1
	setp.eq.f32	%p3, %f18, 0f00000000;
	setp.eq.f32	%p4, %f4, 0f00000000;
	or.pred  	%p5, %p4, %p3;
	.loc 1 31 1
	@%p5 bra 	BB2_6;

	.loc 1 35 1
	mul.f32 	%f24, %f18, %f19;
	.loc 3 3608 3
	div.rn.f32 	%f25, %f4, %f24;
	.loc 1 35 92
	cvt.f64.f32	%fd1, %f25;
	mul.f64 	%fd2, %fd1, 0d3CC7B6EF14E9250C;
	cvt.rn.f32.f64	%f26, %fd2;
	.loc 1 36 1
	mul.f32 	%f27, %f16, %f16;
	.loc 1 37 1
	mul.f32 	%f28, %f15, %f27;
	add.f32 	%f29, %f27, 0f3F800000;
	add.f32 	%f30, %f27, 0fBF800000;
	mul.f32 	%f31, %f12, %f2;
	fma.rn.f32 	%f32, %f11, %f1, %f31;
	fma.rn.f32 	%f33, %f13, %f3, %f32;
	fma.rn.f32 	%f34, %f30, %f33, %f29;
	.loc 3 3608 3
	div.rn.f32 	%f35, %f28, %f34;
	.loc 1 39 1
	mul.f32 	%f36, %f26, %f35;
	.loc 1 40 1
	mul.f32 	%f37, %f26, %f17;
	.loc 1 42 1
	fma.rn.f32 	%f38, %f14, %f14, 0f3F800000;
	rcp.rn.f32 	%f39, %f38;
	.loc 1 43 1
	mul.f32 	%f40, %f14, %f37;
	sub.f32 	%f41, %f36, %f40;
	mul.f32 	%f42, %f39, %f41;
	.loc 1 44 1
	mul.f32 	%f43, %f14, %f36;
	sub.f32 	%f44, %f37, %f43;
	mul.f32 	%f45, %f39, %f44;
	.loc 1 46 1
	mul.f32 	%f46, %f13, %f2;
	mul.f32 	%f47, %f12, %f3;
	sub.f32 	%f48, %f47, %f46;
	mul.f32 	%f49, %f11, %f3;
	mul.f32 	%f50, %f13, %f1;
	sub.f32 	%f51, %f50, %f49;
	mul.f32 	%f52, %f12, %f1;
	mul.f32 	%f53, %f11, %f2;
	sub.f32 	%f54, %f53, %f52;
	.loc 1 47 1
	mul.f32 	%f55, %f2, %f54;
	mul.f32 	%f56, %f3, %f51;
	sub.f32 	%f57, %f55, %f56;
	mul.f32 	%f58, %f3, %f48;
	mul.f32 	%f59, %f1, %f54;
	sub.f32 	%f60, %f58, %f59;
	mul.f32 	%f61, %f1, %f51;
	mul.f32 	%f62, %f2, %f48;
	sub.f32 	%f63, %f61, %f62;
	.loc 1 49 1
	mul.f32 	%f64, %f45, %f48;
	fma.rn.f32 	%f65, %f42, %f57, %f64;
	cvta.to.global.u64 	%rd49, %rd2;
	mul.wide.s32 	%rd50, %r1, 4;
	add.s64 	%rd51, %rd49, %rd50;
	.loc 1 49 1
	ld.global.f32 	%f66, [%rd51];
	add.f32 	%f67, %f66, %f65;
	st.global.f32 	[%rd51], %f67;
	.loc 1 50 1
	mul.f32 	%f68, %f45, %f51;
	fma.rn.f32 	%f69, %f42, %f60, %f68;
	cvta.to.global.u64 	%rd52, %rd3;
	add.s64 	%rd53, %rd52, %rd50;
	.loc 1 50 1
	ld.global.f32 	%f70, [%rd53];
	add.f32 	%f71, %f70, %f69;
	st.global.f32 	[%rd53], %f71;
	.loc 1 51 1
	mul.f32 	%f72, %f45, %f54;
	fma.rn.f32 	%f73, %f42, %f63, %f72;
	cvta.to.global.u64 	%rd54, %rd4;
	add.s64 	%rd55, %rd54, %rd50;
	.loc 1 51 1
	ld.global.f32 	%f74, [%rd55];
	add.f32 	%f75, %f74, %f73;
	st.global.f32 	[%rd55], %f75;

BB2_6:
	.loc 1 53 2
	ret;
}


`
)
