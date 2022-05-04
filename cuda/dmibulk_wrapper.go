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

// CUDA handle for adddmibulk kernel
var adddmibulk_code cu.Function

// Stores the arguments for adddmibulk kernel invocation
type adddmibulk_args_t struct {
	arg_Hx      unsafe.Pointer
	arg_Hy      unsafe.Pointer
	arg_Hz      unsafe.Pointer
	arg_mx      unsafe.Pointer
	arg_my      unsafe.Pointer
	arg_mz      unsafe.Pointer
	arg_Ms_     unsafe.Pointer
	arg_Ms_mul  float32
	arg_aLUT2d  unsafe.Pointer
	arg_DLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_cx      float32
	arg_cy      float32
	arg_cz      float32
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	arg_OpenBC  byte
	argptr      [19]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for adddmibulk kernel invocation
var adddmibulk_args adddmibulk_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	adddmibulk_args.argptr[0] = unsafe.Pointer(&adddmibulk_args.arg_Hx)
	adddmibulk_args.argptr[1] = unsafe.Pointer(&adddmibulk_args.arg_Hy)
	adddmibulk_args.argptr[2] = unsafe.Pointer(&adddmibulk_args.arg_Hz)
	adddmibulk_args.argptr[3] = unsafe.Pointer(&adddmibulk_args.arg_mx)
	adddmibulk_args.argptr[4] = unsafe.Pointer(&adddmibulk_args.arg_my)
	adddmibulk_args.argptr[5] = unsafe.Pointer(&adddmibulk_args.arg_mz)
	adddmibulk_args.argptr[6] = unsafe.Pointer(&adddmibulk_args.arg_Ms_)
	adddmibulk_args.argptr[7] = unsafe.Pointer(&adddmibulk_args.arg_Ms_mul)
	adddmibulk_args.argptr[8] = unsafe.Pointer(&adddmibulk_args.arg_aLUT2d)
	adddmibulk_args.argptr[9] = unsafe.Pointer(&adddmibulk_args.arg_DLUT2d)
	adddmibulk_args.argptr[10] = unsafe.Pointer(&adddmibulk_args.arg_regions)
	adddmibulk_args.argptr[11] = unsafe.Pointer(&adddmibulk_args.arg_cx)
	adddmibulk_args.argptr[12] = unsafe.Pointer(&adddmibulk_args.arg_cy)
	adddmibulk_args.argptr[13] = unsafe.Pointer(&adddmibulk_args.arg_cz)
	adddmibulk_args.argptr[14] = unsafe.Pointer(&adddmibulk_args.arg_Nx)
	adddmibulk_args.argptr[15] = unsafe.Pointer(&adddmibulk_args.arg_Ny)
	adddmibulk_args.argptr[16] = unsafe.Pointer(&adddmibulk_args.arg_Nz)
	adddmibulk_args.argptr[17] = unsafe.Pointer(&adddmibulk_args.arg_PBC)
	adddmibulk_args.argptr[18] = unsafe.Pointer(&adddmibulk_args.arg_OpenBC)
}

// Wrapper for adddmibulk CUDA kernel, asynchronous.
func k_adddmibulk_async(Hx unsafe.Pointer, Hy unsafe.Pointer, Hz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, aLUT2d unsafe.Pointer, DLUT2d unsafe.Pointer, regions unsafe.Pointer, cx float32, cy float32, cz float32, Nx int, Ny int, Nz int, PBC byte, OpenBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("adddmibulk")
	}

	adddmibulk_args.Lock()
	defer adddmibulk_args.Unlock()

	if adddmibulk_code == 0 {
		adddmibulk_code = fatbinLoad(adddmibulk_map, "adddmibulk")
	}

	adddmibulk_args.arg_Hx = Hx
	adddmibulk_args.arg_Hy = Hy
	adddmibulk_args.arg_Hz = Hz
	adddmibulk_args.arg_mx = mx
	adddmibulk_args.arg_my = my
	adddmibulk_args.arg_mz = mz
	adddmibulk_args.arg_Ms_ = Ms_
	adddmibulk_args.arg_Ms_mul = Ms_mul
	adddmibulk_args.arg_aLUT2d = aLUT2d
	adddmibulk_args.arg_DLUT2d = DLUT2d
	adddmibulk_args.arg_regions = regions
	adddmibulk_args.arg_cx = cx
	adddmibulk_args.arg_cy = cy
	adddmibulk_args.arg_cz = cz
	adddmibulk_args.arg_Nx = Nx
	adddmibulk_args.arg_Ny = Ny
	adddmibulk_args.arg_Nz = Nz
	adddmibulk_args.arg_PBC = PBC
	adddmibulk_args.arg_OpenBC = OpenBC

	args := adddmibulk_args.argptr[:]
	cu.LaunchKernel(adddmibulk_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("adddmibulk")
	}
}

// maps compute capability on PTX code for adddmibulk kernel.
var adddmibulk_map = map[int]string{0: "",
	30: adddmibulk_ptx_30}

// adddmibulk PTX code for various compute capabilities.
const (
	adddmibulk_ptx_30 = `
.version 6.4
.target sm_30
.address_size 64

	// .globl	adddmibulk

.visible .entry adddmibulk(
	.param .u64 adddmibulk_param_0,
	.param .u64 adddmibulk_param_1,
	.param .u64 adddmibulk_param_2,
	.param .u64 adddmibulk_param_3,
	.param .u64 adddmibulk_param_4,
	.param .u64 adddmibulk_param_5,
	.param .u64 adddmibulk_param_6,
	.param .f32 adddmibulk_param_7,
	.param .u64 adddmibulk_param_8,
	.param .u64 adddmibulk_param_9,
	.param .u64 adddmibulk_param_10,
	.param .f32 adddmibulk_param_11,
	.param .f32 adddmibulk_param_12,
	.param .f32 adddmibulk_param_13,
	.param .u32 adddmibulk_param_14,
	.param .u32 adddmibulk_param_15,
	.param .u32 adddmibulk_param_16,
	.param .u8 adddmibulk_param_17,
	.param .u8 adddmibulk_param_18
)
{
	.reg .pred 	%p<73>;
	.reg .b16 	%rs<47>;
	.reg .f32 	%f<292>;
	.reg .b32 	%r<238>;
	.reg .b64 	%rd<121>;


	ld.param.u64 	%rd13, [adddmibulk_param_0];
	ld.param.u64 	%rd14, [adddmibulk_param_1];
	ld.param.u64 	%rd15, [adddmibulk_param_2];
	ld.param.u64 	%rd16, [adddmibulk_param_3];
	ld.param.u64 	%rd17, [adddmibulk_param_4];
	ld.param.u64 	%rd18, [adddmibulk_param_5];
	ld.param.u64 	%rd19, [adddmibulk_param_6];
	ld.param.f32 	%f290, [adddmibulk_param_7];
	ld.param.u64 	%rd20, [adddmibulk_param_8];
	ld.param.u64 	%rd21, [adddmibulk_param_9];
	ld.param.u64 	%rd22, [adddmibulk_param_10];
	ld.param.f32 	%f87, [adddmibulk_param_11];
	ld.param.f32 	%f88, [adddmibulk_param_12];
	ld.param.f32 	%f89, [adddmibulk_param_13];
	ld.param.u32 	%r34, [adddmibulk_param_14];
	ld.param.u32 	%r35, [adddmibulk_param_15];
	ld.param.u32 	%r36, [adddmibulk_param_16];
	ld.param.u8 	%rs17, [adddmibulk_param_18];
	ld.param.u8 	%rs16, [adddmibulk_param_17];
	mov.u32 	%r37, %ntid.x;
	mov.u32 	%r38, %ctaid.x;
	mov.u32 	%r39, %tid.x;
	mad.lo.s32 	%r1, %r37, %r38, %r39;
	mov.u32 	%r40, %ntid.y;
	mov.u32 	%r41, %ctaid.y;
	mov.u32 	%r42, %tid.y;
	mad.lo.s32 	%r2, %r40, %r41, %r42;
	mov.u32 	%r43, %ntid.z;
	mov.u32 	%r44, %ctaid.z;
	mov.u32 	%r45, %tid.z;
	mad.lo.s32 	%r3, %r43, %r44, %r45;
	setp.ge.s32	%p1, %r2, %r35;
	setp.ge.s32	%p2, %r1, %r34;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r36;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_62;

	cvta.to.global.u64 	%rd23, %rd22;
	cvta.to.global.u64 	%rd24, %rd18;
	cvta.to.global.u64 	%rd25, %rd17;
	cvta.to.global.u64 	%rd26, %rd16;
	mad.lo.s32 	%r46, %r3, %r35, %r2;
	mad.lo.s32 	%r47, %r46, %r34, %r1;
	mul.wide.s32 	%rd27, %r47, 4;
	add.s64 	%rd28, %rd26, %rd27;
	cvt.s64.s32	%rd29, %r47;
	add.s64 	%rd30, %rd25, %rd27;
	add.s64 	%rd31, %rd24, %rd27;
	add.s64 	%rd32, %rd23, %rd29;
	ld.global.u8 	%rs1, [%rd32];
	ld.global.f32 	%f1, [%rd28];
	ld.global.f32 	%f2, [%rd30];
	mul.f32 	%f90, %f2, %f2;
	fma.rn.f32 	%f91, %f1, %f1, %f90;
	ld.global.f32 	%f3, [%rd31];
	fma.rn.f32 	%f92, %f3, %f3, %f91;
	setp.eq.f32	%p6, %f92, 0f00000000;
	@%p6 bra 	BB0_62;

	and.b16  	%rs18, %rs16, 1;
	setp.eq.b16	%p7, %rs18, 1;
	add.s32 	%r4, %r1, -1;
	@!%p7 bra 	BB0_4;
	bra.uni 	BB0_3;

BB0_3:
	rem.s32 	%r52, %r4, %r34;
	add.s32 	%r53, %r52, %r34;
	rem.s32 	%r232, %r53, %r34;
	bra.uni 	BB0_5;

BB0_4:
	mov.u32 	%r54, 0;
	max.s32 	%r232, %r4, %r54;

BB0_5:
	mad.lo.s32 	%r8, %r46, %r34, %r232;
	setp.eq.b16	%p8, %rs18, 1;
	not.pred 	%p9, %p8;
	setp.lt.s32	%p10, %r4, 0;
	mov.f32 	%f254, 0f00000000;
	and.pred  	%p11, %p10, %p9;
	mov.f32 	%f255, %f254;
	mov.f32 	%f256, %f254;
	@%p11 bra 	BB0_7;

	mul.wide.s32 	%rd34, %r8, 4;
	add.s64 	%rd35, %rd26, %rd34;
	ld.global.f32 	%f254, [%rd35];
	add.s64 	%rd37, %rd25, %rd34;
	ld.global.f32 	%f255, [%rd37];
	add.s64 	%rd39, %rd24, %rd34;
	ld.global.f32 	%f256, [%rd39];

BB0_7:
	mul.f32 	%f96, %f255, %f255;
	fma.rn.f32 	%f97, %f254, %f254, %f96;
	fma.rn.f32 	%f10, %f256, %f256, %f97;
	setp.eq.f32	%p12, %f10, 0f00000000;
	mov.u16 	%rs41, %rs1;
	@%p12 bra 	BB0_9;

	cvt.s64.s32	%rd41, %r8;
	add.s64 	%rd42, %rd23, %rd41;
	ld.global.u8 	%rs41, [%rd42];

BB0_9:
	cvt.u32.u16	%r64, %rs1;
	and.b32  	%r65, %r64, 255;
	setp.gt.u16	%p13, %rs41, %rs1;
	cvt.u32.u16	%r66, %rs41;
	and.b32  	%r67, %r66, 255;
	selp.b32	%r68, %r65, %r67, %p13;
	selp.b32	%r69, %r67, %r65, %p13;
	add.s32 	%r70, %r69, 1;
	mul.lo.s32 	%r71, %r70, %r69;
	shr.u32 	%r72, %r71, 1;
	add.s32 	%r73, %r72, %r68;
	cvta.to.global.u64 	%rd43, %rd20;
	mul.wide.s32 	%rd44, %r73, 4;
	add.s64 	%rd1, %rd43, %rd44;
	cvta.to.global.u64 	%rd45, %rd21;
	add.s64 	%rd2, %rd45, %rd44;
	setp.ne.s16	%p14, %rs17, 0;
	mov.f32 	%f263, 0f00000000;
	and.pred  	%p16, %p12, %p14;
	mov.f32 	%f264, %f263;
	mov.f32 	%f265, %f263;
	@%p16 bra 	BB0_11;

	ld.global.f32 	%f101, [%rd1];
	add.f32 	%f102, %f101, %f101;
	ld.global.f32 	%f103, [%rd2];
	div.rn.f32 	%f104, %f103, %f102;
	mul.f32 	%f105, %f104, %f87;
	fma.rn.f32 	%f106, %f3, %f105, %f2;
	mul.f32 	%f107, %f2, %f105;
	sub.f32 	%f108, %f3, %f107;
	selp.f32	%f109, %f1, %f254, %p12;
	selp.f32	%f110, %f106, %f255, %p12;
	selp.f32	%f111, %f108, %f256, %p12;
	mul.f32 	%f112, %f87, %f87;
	div.rn.f32 	%f113, %f102, %f112;
	sub.f32 	%f114, %f109, %f1;
	sub.f32 	%f115, %f110, %f2;
	sub.f32 	%f116, %f111, %f3;
	fma.rn.f32 	%f263, %f114, %f113, 0f00000000;
	fma.rn.f32 	%f117, %f115, %f113, 0f00000000;
	fma.rn.f32 	%f118, %f116, %f113, 0f00000000;
	div.rn.f32 	%f119, %f103, %f87;
	mul.f32 	%f120, %f111, %f119;
	sub.f32 	%f264, %f117, %f120;
	fma.rn.f32 	%f265, %f110, %f119, %f118;

BB0_11:
	setp.eq.b16	%p18, %rs18, 1;
	add.s32 	%r9, %r1, 1;
	@!%p18 bra 	BB0_13;
	bra.uni 	BB0_12;

BB0_12:
	rem.s32 	%r78, %r9, %r34;
	add.s32 	%r79, %r78, %r34;
	rem.s32 	%r233, %r79, %r34;
	bra.uni 	BB0_14;

BB0_13:
	add.s32 	%r80, %r34, -1;
	min.s32 	%r233, %r9, %r80;

BB0_14:
	setp.eq.b16	%p19, %rs18, 1;
	not.pred 	%p20, %p19;
	mad.lo.s32 	%r13, %r46, %r34, %r233;
	setp.ge.s32	%p21, %r9, %r34;
	mov.f32 	%f260, 0f00000000;
	and.pred  	%p22, %p21, %p20;
	mov.f32 	%f261, %f260;
	mov.f32 	%f262, %f260;
	@%p22 bra 	BB0_16;

	mul.wide.s32 	%rd47, %r13, 4;
	add.s64 	%rd48, %rd26, %rd47;
	ld.global.f32 	%f260, [%rd48];
	add.s64 	%rd50, %rd25, %rd47;
	ld.global.f32 	%f261, [%rd50];
	add.s64 	%rd52, %rd24, %rd47;
	ld.global.f32 	%f262, [%rd52];

BB0_16:
	mul.f32 	%f124, %f261, %f261;
	fma.rn.f32 	%f125, %f260, %f260, %f124;
	fma.rn.f32 	%f23, %f262, %f262, %f125;
	setp.eq.f32	%p23, %f23, 0f00000000;
	mov.u16 	%rs42, %rs1;
	@%p23 bra 	BB0_18;

	cvt.s64.s32	%rd54, %r13;
	add.s64 	%rd55, %rd23, %rd54;
	ld.global.u8 	%rs42, [%rd55];

BB0_18:
	setp.gt.u16	%p24, %rs42, %rs1;
	cvt.u32.u16	%r92, %rs42;
	and.b32  	%r93, %r92, 255;
	selp.b32	%r94, %r65, %r93, %p24;
	selp.b32	%r95, %r93, %r65, %p24;
	add.s32 	%r96, %r95, 1;
	mul.lo.s32 	%r97, %r96, %r95;
	shr.u32 	%r98, %r97, 1;
	add.s32 	%r99, %r98, %r94;
	mul.wide.s32 	%rd57, %r99, 4;
	add.s64 	%rd3, %rd43, %rd57;
	add.s64 	%rd4, %rd45, %rd57;
	and.pred  	%p27, %p23, %p14;
	@%p27 bra 	BB0_20;

	ld.global.f32 	%f126, [%rd3];
	add.f32 	%f127, %f126, %f126;
	ld.global.f32 	%f128, [%rd4];
	div.rn.f32 	%f129, %f128, %f127;
	mul.f32 	%f130, %f129, %f87;
	mul.f32 	%f131, %f3, %f130;
	sub.f32 	%f132, %f2, %f131;
	fma.rn.f32 	%f133, %f2, %f130, %f3;
	selp.f32	%f134, %f1, %f260, %p23;
	selp.f32	%f135, %f132, %f261, %p23;
	selp.f32	%f136, %f133, %f262, %p23;
	mul.f32 	%f137, %f87, %f87;
	div.rn.f32 	%f138, %f127, %f137;
	sub.f32 	%f139, %f134, %f1;
	sub.f32 	%f140, %f135, %f2;
	sub.f32 	%f141, %f136, %f3;
	fma.rn.f32 	%f263, %f139, %f138, %f263;
	fma.rn.f32 	%f142, %f140, %f138, %f264;
	fma.rn.f32 	%f143, %f141, %f138, %f265;
	div.rn.f32 	%f144, %f128, %f87;
	fma.rn.f32 	%f264, %f136, %f144, %f142;
	mul.f32 	%f145, %f135, %f144;
	sub.f32 	%f265, %f143, %f145;

BB0_20:
	and.b16  	%rs6, %rs16, 2;
	setp.eq.s16	%p29, %rs6, 0;
	add.s32 	%r14, %r2, -1;
	@%p29 bra 	BB0_22;

	rem.s32 	%r104, %r14, %r35;
	add.s32 	%r105, %r104, %r35;
	rem.s32 	%r234, %r105, %r35;
	bra.uni 	BB0_23;

BB0_22:
	mov.u32 	%r106, 0;
	max.s32 	%r234, %r14, %r106;

BB0_23:
	mad.lo.s32 	%r111, %r3, %r35, %r234;
	mad.lo.s32 	%r18, %r111, %r34, %r1;
	setp.lt.s32	%p31, %r14, 0;
	mov.f32 	%f266, 0f00000000;
	and.pred  	%p32, %p31, %p29;
	mov.f32 	%f267, %f266;
	mov.f32 	%f268, %f266;
	@%p32 bra 	BB0_25;

	mul.wide.s32 	%rd60, %r18, 4;
	add.s64 	%rd61, %rd26, %rd60;
	ld.global.f32 	%f266, [%rd61];
	add.s64 	%rd63, %rd25, %rd60;
	ld.global.f32 	%f267, [%rd63];
	add.s64 	%rd65, %rd24, %rd60;
	ld.global.f32 	%f268, [%rd65];

BB0_25:
	mul.f32 	%f149, %f267, %f267;
	fma.rn.f32 	%f150, %f266, %f266, %f149;
	fma.rn.f32 	%f36, %f268, %f268, %f150;
	setp.eq.f32	%p33, %f36, 0f00000000;
	mov.u16 	%rs43, %rs1;
	@%p33 bra 	BB0_27;

	cvt.s64.s32	%rd67, %r18;
	add.s64 	%rd68, %rd23, %rd67;
	ld.global.u8 	%rs43, [%rd68];

BB0_27:
	setp.gt.u16	%p34, %rs43, %rs1;
	cvt.u32.u16	%r118, %rs43;
	and.b32  	%r119, %r118, 255;
	selp.b32	%r120, %r65, %r119, %p34;
	selp.b32	%r121, %r119, %r65, %p34;
	add.s32 	%r122, %r121, 1;
	mul.lo.s32 	%r123, %r122, %r121;
	shr.u32 	%r124, %r123, 1;
	add.s32 	%r125, %r124, %r120;
	mul.wide.s32 	%rd70, %r125, 4;
	add.s64 	%rd5, %rd43, %rd70;
	add.s64 	%rd6, %rd45, %rd70;
	and.pred  	%p37, %p33, %p14;
	@%p37 bra 	BB0_29;

	ld.global.f32 	%f151, [%rd5];
	add.f32 	%f152, %f151, %f151;
	ld.global.f32 	%f153, [%rd6];
	div.rn.f32 	%f154, %f153, %f152;
	mul.f32 	%f155, %f154, %f88;
	mul.f32 	%f156, %f3, %f155;
	sub.f32 	%f157, %f1, %f156;
	fma.rn.f32 	%f158, %f1, %f155, %f3;
	selp.f32	%f159, %f157, %f266, %p33;
	selp.f32	%f160, %f2, %f267, %p33;
	selp.f32	%f161, %f158, %f268, %p33;
	mul.f32 	%f162, %f88, %f88;
	div.rn.f32 	%f163, %f152, %f162;
	sub.f32 	%f164, %f159, %f1;
	sub.f32 	%f165, %f160, %f2;
	sub.f32 	%f166, %f161, %f3;
	fma.rn.f32 	%f167, %f164, %f163, %f263;
	fma.rn.f32 	%f264, %f165, %f163, %f264;
	fma.rn.f32 	%f168, %f166, %f163, %f265;
	div.rn.f32 	%f169, %f153, %f88;
	fma.rn.f32 	%f263, %f161, %f169, %f167;
	mul.f32 	%f170, %f159, %f169;
	sub.f32 	%f265, %f168, %f170;

BB0_29:
	add.s32 	%r19, %r2, 1;
	@%p29 bra 	BB0_31;

	rem.s32 	%r130, %r19, %r35;
	add.s32 	%r131, %r130, %r35;
	rem.s32 	%r235, %r131, %r35;
	bra.uni 	BB0_32;

BB0_31:
	add.s32 	%r132, %r35, -1;
	min.s32 	%r235, %r19, %r132;

BB0_32:
	shr.u16 	%rs30, %rs16, 1;
	and.b16  	%rs31, %rs30, 1;
	setp.eq.b16	%p40, %rs31, 1;
	not.pred 	%p41, %p40;
	mad.lo.s32 	%r137, %r3, %r35, %r235;
	mad.lo.s32 	%r23, %r137, %r34, %r1;
	setp.ge.s32	%p42, %r19, %r35;
	mov.f32 	%f272, 0f00000000;
	and.pred  	%p43, %p42, %p41;
	mov.f32 	%f273, %f272;
	mov.f32 	%f274, %f272;
	@%p43 bra 	BB0_34;

	mul.wide.s32 	%rd73, %r23, 4;
	add.s64 	%rd74, %rd26, %rd73;
	ld.global.f32 	%f272, [%rd74];
	add.s64 	%rd76, %rd25, %rd73;
	ld.global.f32 	%f273, [%rd76];
	add.s64 	%rd78, %rd24, %rd73;
	ld.global.f32 	%f274, [%rd78];

BB0_34:
	mul.f32 	%f174, %f273, %f273;
	fma.rn.f32 	%f175, %f272, %f272, %f174;
	fma.rn.f32 	%f49, %f274, %f274, %f175;
	setp.eq.f32	%p44, %f49, 0f00000000;
	mov.u16 	%rs44, %rs1;
	@%p44 bra 	BB0_36;

	cvt.s64.s32	%rd80, %r23;
	add.s64 	%rd81, %rd23, %rd80;
	ld.global.u8 	%rs44, [%rd81];

BB0_36:
	setp.gt.u16	%p45, %rs44, %rs1;
	cvt.u32.u16	%r144, %rs44;
	and.b32  	%r145, %r144, 255;
	selp.b32	%r146, %r65, %r145, %p45;
	selp.b32	%r147, %r145, %r65, %p45;
	add.s32 	%r148, %r147, 1;
	mul.lo.s32 	%r149, %r148, %r147;
	shr.u32 	%r150, %r149, 1;
	add.s32 	%r151, %r150, %r146;
	mul.wide.s32 	%rd83, %r151, 4;
	add.s64 	%rd7, %rd43, %rd83;
	add.s64 	%rd8, %rd45, %rd83;
	and.pred  	%p48, %p44, %p14;
	@%p48 bra 	BB0_38;

	ld.global.f32 	%f176, [%rd7];
	add.f32 	%f177, %f176, %f176;
	ld.global.f32 	%f178, [%rd8];
	div.rn.f32 	%f179, %f178, %f177;
	mul.f32 	%f180, %f179, %f88;
	fma.rn.f32 	%f181, %f3, %f180, %f1;
	mul.f32 	%f182, %f1, %f180;
	sub.f32 	%f183, %f3, %f182;
	selp.f32	%f184, %f181, %f272, %p44;
	selp.f32	%f185, %f2, %f273, %p44;
	selp.f32	%f186, %f183, %f274, %p44;
	mul.f32 	%f187, %f88, %f88;
	div.rn.f32 	%f188, %f177, %f187;
	sub.f32 	%f189, %f184, %f1;
	sub.f32 	%f190, %f185, %f2;
	sub.f32 	%f191, %f186, %f3;
	fma.rn.f32 	%f192, %f189, %f188, %f263;
	fma.rn.f32 	%f264, %f190, %f188, %f264;
	fma.rn.f32 	%f193, %f191, %f188, %f265;
	div.rn.f32 	%f194, %f178, %f88;
	mul.f32 	%f195, %f186, %f194;
	sub.f32 	%f263, %f192, %f195;
	fma.rn.f32 	%f265, %f184, %f194, %f193;

BB0_38:
	setp.eq.s32	%p50, %r36, 1;
	@%p50 bra 	BB0_57;

	and.b16  	%rs11, %rs16, 4;
	setp.eq.s16	%p51, %rs11, 0;
	add.s32 	%r24, %r3, -1;
	@%p51 bra 	BB0_41;

	rem.s32 	%r156, %r24, %r36;
	add.s32 	%r157, %r156, %r36;
	rem.s32 	%r236, %r157, %r36;
	bra.uni 	BB0_42;

BB0_41:
	mov.u32 	%r158, 0;
	max.s32 	%r236, %r24, %r158;

BB0_42:
	mad.lo.s32 	%r163, %r236, %r35, %r2;
	mad.lo.s32 	%r28, %r163, %r34, %r1;
	setp.lt.s32	%p53, %r24, 0;
	mov.f32 	%f278, 0f00000000;
	and.pred  	%p54, %p53, %p51;
	mov.f32 	%f279, %f278;
	mov.f32 	%f280, %f278;
	@%p54 bra 	BB0_44;

	mul.wide.s32 	%rd86, %r28, 4;
	add.s64 	%rd87, %rd26, %rd86;
	ld.global.f32 	%f278, [%rd87];
	add.s64 	%rd89, %rd25, %rd86;
	ld.global.f32 	%f279, [%rd89];
	add.s64 	%rd91, %rd24, %rd86;
	ld.global.f32 	%f280, [%rd91];

BB0_44:
	mul.f32 	%f199, %f279, %f279;
	fma.rn.f32 	%f200, %f278, %f278, %f199;
	fma.rn.f32 	%f62, %f280, %f280, %f200;
	setp.eq.f32	%p55, %f62, 0f00000000;
	mov.u16 	%rs45, %rs1;
	@%p55 bra 	BB0_46;

	cvt.s64.s32	%rd93, %r28;
	add.s64 	%rd94, %rd23, %rd93;
	ld.global.u8 	%rs45, [%rd94];

BB0_46:
	setp.gt.u16	%p56, %rs45, %rs1;
	cvt.u32.u16	%r170, %rs45;
	and.b32  	%r171, %r170, 255;
	selp.b32	%r172, %r65, %r171, %p56;
	selp.b32	%r173, %r171, %r65, %p56;
	add.s32 	%r174, %r173, 1;
	mul.lo.s32 	%r175, %r174, %r173;
	shr.u32 	%r176, %r175, 1;
	add.s32 	%r177, %r176, %r172;
	mul.wide.s32 	%rd96, %r177, 4;
	add.s64 	%rd9, %rd43, %rd96;
	add.s64 	%rd10, %rd45, %rd96;
	and.pred  	%p59, %p55, %p14;
	@%p59 bra 	BB0_48;

	ld.global.f32 	%f201, [%rd9];
	add.f32 	%f202, %f201, %f201;
	ld.global.f32 	%f203, [%rd10];
	div.rn.f32 	%f204, %f203, %f202;
	mul.f32 	%f205, %f204, %f89;
	fma.rn.f32 	%f206, %f2, %f205, %f1;
	mul.f32 	%f207, %f1, %f205;
	sub.f32 	%f208, %f2, %f207;
	selp.f32	%f209, %f206, %f278, %p55;
	selp.f32	%f210, %f208, %f279, %p55;
	selp.f32	%f211, %f3, %f280, %p55;
	mul.f32 	%f212, %f89, %f89;
	div.rn.f32 	%f213, %f202, %f212;
	sub.f32 	%f214, %f209, %f1;
	sub.f32 	%f215, %f210, %f2;
	sub.f32 	%f216, %f211, %f3;
	fma.rn.f32 	%f217, %f214, %f213, %f263;
	fma.rn.f32 	%f218, %f215, %f213, %f264;
	fma.rn.f32 	%f265, %f216, %f213, %f265;
	div.rn.f32 	%f219, %f203, %f89;
	mul.f32 	%f220, %f210, %f219;
	sub.f32 	%f263, %f217, %f220;
	fma.rn.f32 	%f264, %f209, %f219, %f218;

BB0_48:
	add.s32 	%r29, %r3, 1;
	@%p51 bra 	BB0_50;

	rem.s32 	%r182, %r29, %r36;
	add.s32 	%r183, %r182, %r36;
	rem.s32 	%r237, %r183, %r36;
	bra.uni 	BB0_51;

BB0_50:
	add.s32 	%r184, %r36, -1;
	min.s32 	%r237, %r29, %r184;

BB0_51:
	mad.lo.s32 	%r189, %r237, %r35, %r2;
	mad.lo.s32 	%r33, %r189, %r34, %r1;
	setp.ge.s32	%p62, %r29, %r36;
	mov.f32 	%f284, 0f00000000;
	and.pred  	%p64, %p62, %p51;
	mov.f32 	%f285, %f284;
	mov.f32 	%f286, %f284;
	@%p64 bra 	BB0_53;

	mul.wide.s32 	%rd99, %r33, 4;
	add.s64 	%rd100, %rd26, %rd99;
	ld.global.f32 	%f286, [%rd100];
	add.s64 	%rd102, %rd25, %rd99;
	ld.global.f32 	%f285, [%rd102];
	add.s64 	%rd104, %rd24, %rd99;
	ld.global.f32 	%f284, [%rd104];

BB0_53:
	mul.f32 	%f224, %f286, %f286;
	fma.rn.f32 	%f225, %f285, %f285, %f224;
	fma.rn.f32 	%f75, %f284, %f284, %f225;
	setp.eq.f32	%p65, %f75, 0f00000000;
	mov.u16 	%rs46, %rs1;
	@%p65 bra 	BB0_55;

	cvt.s64.s32	%rd106, %r33;
	add.s64 	%rd107, %rd23, %rd106;
	ld.global.u8 	%rs46, [%rd107];

BB0_55:
	setp.gt.u16	%p66, %rs46, %rs1;
	cvt.u32.u16	%r196, %rs46;
	and.b32  	%r197, %r196, 255;
	selp.b32	%r198, %r65, %r197, %p66;
	selp.b32	%r199, %r197, %r65, %p66;
	add.s32 	%r200, %r199, 1;
	mul.lo.s32 	%r201, %r200, %r199;
	shr.u32 	%r202, %r201, 1;
	add.s32 	%r203, %r202, %r198;
	mul.wide.s32 	%rd109, %r203, 4;
	add.s64 	%rd11, %rd43, %rd109;
	add.s64 	%rd12, %rd45, %rd109;
	and.pred  	%p69, %p65, %p14;
	@%p69 bra 	BB0_57;

	ld.global.f32 	%f226, [%rd11];
	add.f32 	%f227, %f226, %f226;
	ld.global.f32 	%f228, [%rd12];
	div.rn.f32 	%f229, %f228, %f227;
	mul.f32 	%f230, %f229, %f89;
	mul.f32 	%f231, %f2, %f230;
	sub.f32 	%f232, %f1, %f231;
	fma.rn.f32 	%f233, %f1, %f230, %f2;
	selp.f32	%f234, %f3, %f284, %p65;
	selp.f32	%f235, %f233, %f285, %p65;
	selp.f32	%f236, %f232, %f286, %p65;
	mul.f32 	%f237, %f89, %f89;
	div.rn.f32 	%f238, %f227, %f237;
	sub.f32 	%f239, %f236, %f1;
	sub.f32 	%f240, %f235, %f2;
	sub.f32 	%f241, %f234, %f3;
	fma.rn.f32 	%f242, %f239, %f238, %f263;
	fma.rn.f32 	%f243, %f240, %f238, %f264;
	fma.rn.f32 	%f265, %f241, %f238, %f265;
	div.rn.f32 	%f244, %f228, %f89;
	fma.rn.f32 	%f263, %f235, %f244, %f242;
	mul.f32 	%f245, %f236, %f244;
	sub.f32 	%f264, %f243, %f245;

BB0_57:
	setp.eq.s64	%p71, %rd19, 0;
	@%p71 bra 	BB0_59;

	cvta.to.global.u64 	%rd111, %rd19;
	add.s64 	%rd113, %rd111, %rd27;
	ld.global.f32 	%f246, [%rd113];
	mul.f32 	%f290, %f246, %f290;

BB0_59:
	setp.eq.f32	%p72, %f290, 0f00000000;
	mov.f32 	%f291, 0f00000000;
	@%p72 bra 	BB0_61;

	rcp.rn.f32 	%f291, %f290;

BB0_61:
	cvta.to.global.u64 	%rd114, %rd13;
	add.s64 	%rd116, %rd114, %rd27;
	ld.global.f32 	%f248, [%rd116];
	fma.rn.f32 	%f249, %f263, %f291, %f248;
	st.global.f32 	[%rd116], %f249;
	cvta.to.global.u64 	%rd117, %rd14;
	add.s64 	%rd118, %rd117, %rd27;
	ld.global.f32 	%f250, [%rd118];
	fma.rn.f32 	%f251, %f264, %f291, %f250;
	st.global.f32 	[%rd118], %f251;
	cvta.to.global.u64 	%rd119, %rd15;
	add.s64 	%rd120, %rd119, %rd27;
	ld.global.f32 	%f252, [%rd120];
	fma.rn.f32 	%f253, %f265, %f291, %f252;
	st.global.f32 	[%rd120], %f253;

BB0_62:
	ret;
}


`
)
