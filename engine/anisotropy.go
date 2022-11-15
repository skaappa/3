package engine

// Magnetocrystalline anisotropy.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
)

// Anisotropy variables
var (
	Ku1        = NewScalarParam("Ku1", "J/m3", "1st order uniaxial anisotropy constant")
	Ku2        = NewScalarParam("Ku2", "J/m3", "2nd order uniaxial anisotropy constant")
	Kc1        = NewScalarParam("Kc1", "J/m3", "1st order cubic anisotropy constant")
	Kc2        = NewScalarParam("Kc2", "J/m3", "2nd order cubic anisotropy constant")
	Kc3        = NewScalarParam("Kc3", "J/m3", "3rd order cubic anisotropy constant")
	Kt1        = NewShiftableField("Kt1", "J/m3", "1st axis triaxial anisotropy constant", 1, setKt1)
	Kt2        = NewShiftableField("Kt2", "J/m3", "2nd axis triaxial anisotropy constant", 1, setKt2)
	Kt3        = NewShiftableField("Kt3", "J/m3", "3rd axis triaxial anisotropy constant", 1, setKt3)
	AnisU      = NewVectorParam("anisU", "", "Uniaxial anisotropy direction")
	AnisC1     = NewVectorParam("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2     = NewVectorParam("anisC2", "", "Cubic anisotorpy directon #2")
	AnisT1     = NewShiftableField("anisT1", "", "Triaxial anisotropy direction #1", 3, setAnisT1)
	AnisT2     = NewShiftableField("anisT2", "", "Triaxial anisotropy direction #2", 3, setAnisT2)
	AnisT3     = NewShiftableField("anisT3", "", "Triaxial anisotropy direction #3", 3, setAnisT3)
	B_anis     = NewVectorField("B_anis", "T", "Anisotropy field", AddAnisotropyField)
	Edens_anis = NewScalarField("Edens_anis", "J/m3", "Anisotropy energy density", AddAnisotropyEnergyDensity)
	E_anis     = NewScalarValue("E_anis", "J", "total anisotropy energy", GetAnisotropyEnergy)
)

var (
	sZero = NewScalarParam("_zero", "", "utility zero parameter")
        sZeroScalar = NewShiftableField("_zerofield", "", "utility zero scalar field", 3, setZeroField)
)

var (
    saved = false
    kt1fromfile = LoadFile("mykt1.ovf")
    kt2fromfile = LoadFile("mykt2.ovf")
    kt3fromfile = LoadFile("mykt3.ovf")
    anist1fromfile = LoadFile("myanist1.ovf")
    anist2fromfile = LoadFile("myanist2.ovf")
    anist3fromfile = LoadFile("myanist3.ovf")
)

func init() {
	registerEnergy(GetAnisotropyEnergy, AddAnisotropyEnergyDensity)
}


func setZeroField(*data.Slice) {
}

func setKt1(dst *data.Slice) {
    LoadKt1(dst)
}

func setKt2(dst *data.Slice) {
    LoadKt2(dst)
}

func setKt3(dst *data.Slice) {
    LoadKt3(dst)
}

func setAnisT1(dst *data.Slice) {
    LoadAnisT1(dst)
}

func setAnisT2(dst *data.Slice) {
    LoadAnisT2(dst)
}

func setAnisT3(dst *data.Slice) {
    LoadAnisT3(dst)
}

func LoadKt1(dst *data.Slice) {
     kt1 := cuda.NewSlice(1, Mesh().Size())
     data.Copy(kt1, kt1fromfile)
     data.Copy(dst, kt1)
     kt1.Free()
}

func LoadKt2(dst *data.Slice) {
     kt2 := cuda.NewSlice(1, Mesh().Size())
     data.Copy(kt2, kt2fromfile)
     data.Copy(dst, kt2)
     kt2.Free()
}

func LoadKt3(dst *data.Slice) {
     kt3 := cuda.NewSlice(1, Mesh().Size())
     data.Copy(kt3, kt3fromfile)
     data.Copy(dst, kt3)
     kt3.Free()
}

func LoadAnisT1(dst *data.Slice) {
     anist1 := cuda.NewSlice(3, Mesh().Size())
     data.Copy(anist1, anist1fromfile)
     data.Copy(dst, anist1)
     anist1.Free()
}

func LoadAnisT2(dst *data.Slice) {
     anist2 := cuda.NewSlice(3, Mesh().Size())
     data.Copy(anist2, anist2fromfile)
     data.Copy(dst, anist2)
     anist2.Free()
}

func LoadAnisT3(dst *data.Slice) {
     anist3 := cuda.NewSlice(3, Mesh().Size())
     data.Copy(anist3, anist3fromfile)
     data.Copy(dst, anist3)
     anist3.Free()
}

func mysave(fname string, sfield ShiftableField) {
     if !saved {
         s := sfield.buffer.HostCopy()
         f, err := httpfs.Create(fname)
         util.FatalErr(err)
         defer f.Close()
         info := data.Meta{Time: 0., Name: "myanis", Unit: "", CellSize: Mesh().CellSize()}
         oommf.WriteOVF2(f, s, info, "text")
         // saved = true
     }
}

func addUniaxialAnisotropyFrom(dst *data.Slice, M magnetization, Msat, Ku1, Ku2 *RegionwiseScalar, AnisU *RegionwiseVector) {
	if Ku1.nonZero() || Ku2.nonZero() {

		ms := Msat.MSlice()
		defer ms.Recycle()

		ku1 := Ku1.MSlice()
		defer ku1.Recycle()

		ku2 := Ku2.MSlice()
		defer ku2.Recycle()

		u := AnisU.MSlice()
		defer u.Recycle()

		cuda.AddUniaxialAnisotropy2(dst, M.Buffer(), ms, ku1, ku2, u)
 	}
}

func addCubicAnisotropyFrom(dst *data.Slice, M magnetization, Msat, Kc1, Kc2, Kc3 *RegionwiseScalar, AnisC1, AnisC2 *RegionwiseVector) {
	if Kc1.nonZero() || Kc2.nonZero() || Kc3.nonZero() {
		ms := Msat.MSlice()
		defer ms.Recycle()

		kc1 := Kc1.MSlice()
		defer kc1.Recycle()

		kc2 := Kc2.MSlice()
		defer kc2.Recycle()

		kc3 := Kc3.MSlice()
		defer kc3.Recycle()

		c1 := AnisC1.MSlice()
		defer c1.Recycle()

		c2 := AnisC2.MSlice()
		defer c2.Recycle()
		cuda.AddCubicAnisotropy2(dst, M.Buffer(), ms, kc1, kc2, kc3, c1, c2)
	}
}

func addTriaxialAnisotropyFrom(dst *data.Slice, M magnetization, Msat *RegionwiseScalar, Kt1, Kt2, Kt3, AnisT1, AnisT2, AnisT3 ShiftableField) {
	// if Kt1.nonZero() || Kt2.nonZero() || Kt3.nonZero() {
		ms := Msat.MSlice()
		defer ms.Recycle()

		// Kt1.init()
		kt1 := cuda.ToMSlice(Kt1.Buffer())
		defer kt1.Recycle()

		kt2 := cuda.ToMSlice(Kt2.Buffer())
		defer kt2.Recycle()

		kt3 := cuda.ToMSlice(Kt3.buffer)
		defer kt3.Recycle()

		t1 := cuda.ToMSlice(AnisT1.buffer)
		defer t1.Recycle()

		t2 := cuda.ToMSlice(AnisT2.buffer)
		defer t2.Recycle()

		t3 := cuda.ToMSlice(AnisT3.buffer)
		defer t3.Recycle()

		cuda.AddTriaxialAnisotropy2(dst, M.Buffer(), ms, kt1, kt2, kt3, t1, t2, t3)
	// }
}


// Add the anisotropy field to dst
func AddAnisotropyField(dst *data.Slice) {
	addUniaxialAnisotropyFrom(dst, M, Msat, Ku1, Ku2, AnisU)
	addCubicAnisotropyFrom(dst, M, Msat, Kc1, Kc2, Kc3, AnisC1, AnisC2)
	addTriaxialAnisotropyFrom(dst, M, Msat, Kt1, Kt2, Kt3, AnisT1, AnisT2, AnisT3)
}

// Add the anisotropy energy density to dst
func AddAnisotropyEnergyDensity(dst *data.Slice) {
	haveUniaxial := Ku1.nonZero() || Ku2.nonZero()
	haveCubic := Kc1.nonZero() || Kc2.nonZero() || Kc3.nonZero()
	haveTriaxial := true  // Kt1.nonZero() || Kt2.nonZero() || Kt3.nonZero()	

	if !haveUniaxial && !haveCubic {
	        if !haveTriaxial {
		        return
		}
	}

	buf := cuda.Buffer(B_anis.NComp(), Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf := ValueOf(M_full)
	defer cuda.Recycle(Mf)

	if haveUniaxial {
		// 1st
		cuda.Zero(buf)
		addUniaxialAnisotropyFrom(buf, M, Msat, Ku1, sZero, AnisU)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)

		// 2nd
		cuda.Zero(buf)
		addUniaxialAnisotropyFrom(buf, M, Msat, sZero, Ku2, AnisU)
		cuda.AddDotProduct(dst, -1./4., buf, Mf)
	}

	if haveCubic {
		// 1st
		cuda.Zero(buf)
		addCubicAnisotropyFrom(buf, M, Msat, Kc1, sZero, sZero, AnisC1, AnisC2)
		cuda.AddDotProduct(dst, -1./4., buf, Mf)

		// 2nd
		cuda.Zero(buf)
		addCubicAnisotropyFrom(buf, M, Msat, sZero, Kc2, sZero, AnisC1, AnisC2)
		cuda.AddDotProduct(dst, -1./6., buf, Mf)

		// 3nd
		cuda.Zero(buf)
		addCubicAnisotropyFrom(buf, M, Msat, sZero, sZero, Kc3, AnisC1, AnisC2)
		cuda.AddDotProduct(dst, -1./8., buf, Mf)
	}


	if haveTriaxial {
		// 1st axis
		cuda.Zero(buf)
		addTriaxialAnisotropyFrom(buf, M, Msat, Kt1, sZeroScalar, sZeroScalar, AnisT1, AnisT2, AnisT3)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)

		// 2nd
		cuda.Zero(buf)
		addTriaxialAnisotropyFrom(buf, M, Msat, sZeroScalar, Kt2, sZeroScalar, AnisT1, AnisT2, AnisT3)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)

		// 3nd
		cuda.Zero(buf)
		addTriaxialAnisotropyFrom(buf, M, Msat, sZeroScalar, sZeroScalar, Kt3, AnisT1, AnisT2, AnisT3)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)
	}
}

// Returns anisotropy energy in joules.
func GetAnisotropyEnergy() float64 {
	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddAnisotropyEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}
