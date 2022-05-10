package engine

// Magnetocrystalline anisotropy.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
        "log"
)

// Anisotropy variables
var (
	Ku2        = NewScalarParam("Ku2", "J/m3", "2nd order uniaxial anisotropy constant")
	Ku1        = NewScalarField("Ku1", "J/m3", "1st order uniaxial anisotropy constant", nullmethod)

	Kc1        = NewScalarParam("Kc1", "J/m3", "1st order cubic anisotropy constant")
	Kc2        = NewScalarParam("Kc2", "J/m3", "2nd order cubic anisotropy constant")
	Kc3        = NewScalarParam("Kc3", "J/m3", "3rd order cubic anisotropy constant")
	Kt1        = NewScalarParam("Kt1", "J/m3", "1st axis triaxial anisotropy constant")
	Kt2        = NewScalarParam("Kt2", "J/m3", "2nd axis triaxial anisotropy constant")
	Kt3        = NewScalarParam("Kt3", "J/m3", "3rd axis triaxial anisotropy constant")
	AnisU      = NewVectorField("anisU", "", "Uniaxial anisotropy direction", nullmethod)
	AnisC1     = NewVectorParam("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2     = NewVectorParam("anisC2", "", "Cubic anisotorpy directon #2")
	AnisT1     = NewVectorParam("anisT1", "", "Triaxial anisotropy direction #1")
	AnisT2     = NewVectorParam("anisT2", "", "Triaxial anisotropy direction #2")
	AnisT3     = NewVectorParam("anisT3", "", "Triaxial anisotropy direction #3")
	B_anis     = NewVectorField("B_anis", "T", "Anisotropy field", AddAnisotropyField)
	Edens_anis = NewScalarField("Edens_anis", "J/m3", "Anisotropy energy density", AddAnisotropyEnergyDensity)
	E_anis     = NewScalarValue("E_anis", "J", "total anisotropy energy", GetAnisotropyEnergy)
)

var (
	sZero = NewScalarParam("_zero", "", "utility zero parameter")
)

var (
    saved = false
)

func init() {
	registerEnergy(GetAnisotropyEnergy, AddAnisotropyEnergyDensity)
}


func nullmethod(*data.Slice) {
}

func LoadKu1() *data.Slice{
    // filedata := LoadFile("./Ku1.ovf")
    size := Mesh().Size()
    Ku1slice := cuda.NewSlice(1, size)
    for i:=0; i<size[0]; i+=1 {
        for j:=0; j<size[1]; j+=1 {
           cuda.SetCell(Ku1slice, 0, i, j, 0, 80000.)
	  }
      }
    
    return Ku1slice
}

func LoadAnisU() *data.Slice {
    size := Mesh().Size()
    AnisUslice := cuda.NewSlice(3, size)
    for i:=0; i<size[0]; i+=1 {
        for j:=0; j<size[1]; j+=1 {
           cuda.SetCell(AnisUslice, 0, i, j, 0, float32(i)*float32(i))
           cuda.SetCell(AnisUslice, 1, i, j, 0, float32(j)*float32(j))
	  }
      }
    
    return AnisUslice

}

func mysave(fname string, s *data.Slice) {
     if !saved {
         f, err := httpfs.Create(fname)
         util.FatalErr(err)
         defer f.Close()
         info := data.Meta{Time: 0., Name: "myanis", Unit: "", CellSize: Mesh().CellSize()}
         oommf.WriteOVF1(f, s, info, "text")
         saved = true
     }
}

func addUniaxialAnisotropyFrom(dst *data.Slice, M magnetization, Msat, Ku2 *RegionwiseScalar, Ku1 ScalarField, AnisU VectorField) {
	// if Ku1.nonZero() || Ku2.nonZero() {
	        log.Println("here 1")

		ms := Msat.MSlice()
		defer ms.Recycle()

                ku1slice := LoadKu1()
                ku1 := cuda.ToMSlice(ku1slice)
		// defer ku1.Recycle()

		ku2 := Ku2.MSlice()
		defer ku2.Recycle()

                uslice := LoadAnisU()
                mysave("anisu.ovf", uslice.HostCopy())
                u := cuda.ToMSlice(uslice)
		// defer u.Recycle()

	        log.Println("here  2")

		cuda.AddUniaxialAnisotropy2(dst, M.Buffer(), ms, ku1, ku2, u)
	        log.Println("here   3")
	// }
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

func addTriaxialAnisotropyFrom(dst *data.Slice, M magnetization, Msat, Kt1, Kt2, Kt3 *RegionwiseScalar, AnisT1, AnisT2, AnisT3 *RegionwiseVector) {
	if Kt1.nonZero() || Kt2.nonZero() || Kt3.nonZero() {
		ms := Msat.MSlice()
		defer ms.Recycle()

		kt1 := Kt1.MSlice()
		defer kt1.Recycle()

		kt2 := Kt2.MSlice()
		defer kt2.Recycle()

		kt3 := Kt3.MSlice()
		defer kt3.Recycle()

		t1 := AnisT1.MSlice()
		defer t1.Recycle()

		t2 := AnisT2.MSlice()
		defer t2.Recycle()

		t3 := AnisT3.MSlice()
		defer t3.Recycle()
		
		cuda.AddTriaxialAnisotropy2(dst, M.Buffer(), ms, kt1, kt2, kt3, t1, t2, t3)
	}
}


// Add the anisotropy field to dst
func AddAnisotropyField(dst *data.Slice) {
	addUniaxialAnisotropyFrom(dst, M, Msat, Ku2, Ku1, AnisU)
	addCubicAnisotropyFrom(dst, M, Msat, Kc1, Kc2, Kc3, AnisC1, AnisC2)
	addTriaxialAnisotropyFrom(dst, M, Msat, Kt1, Kt2, Kt3, AnisT1, AnisT2, AnisT3)
}

// Add the anisotropy energy density to dst
func AddAnisotropyEnergyDensity(dst *data.Slice) {
	haveUniaxial := true  // Ku1.nonZero() || Ku2.nonZero()
	haveCubic := Kc1.nonZero() || Kc2.nonZero() || Kc3.nonZero()
	haveTriaxial := Kt1.nonZero() || Kt2.nonZero() || Kt3.nonZero()	

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
		addUniaxialAnisotropyFrom(buf, M, Msat, sZero, Ku1, AnisU)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)
                log.Println("here    4")

		// 2nd
		// cuda.Zero(buf)
		// addUniaxialAnisotropyFrom(buf, M, Msat, sZero, Ku2, AnisU)
		// cuda.AddDotProduct(dst, -1./4., buf, Mf)
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
		addTriaxialAnisotropyFrom(buf, M, Msat, Kt1, sZero, sZero, AnisT1, AnisT2, AnisT3)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)

		// 2nd
		cuda.Zero(buf)
		addTriaxialAnisotropyFrom(buf, M, Msat, sZero, Kt2, sZero, AnisT1, AnisT2, AnisT3)
		cuda.AddDotProduct(dst, -1./2., buf, Mf)

		// 3nd
		cuda.Zero(buf)
		addTriaxialAnisotropyFrom(buf, M, Msat, sZero, sZero, Kt3, AnisT1, AnisT2, AnisT3)
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
