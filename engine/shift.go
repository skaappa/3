package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	TotalShift, TotalYShift                    float64                        // accumulated window shift (X and Y) in meter
	ShiftMagL, ShiftMagR, ShiftMagU, ShiftMagD data.Vector                    // when shifting m, put these value at the left/right edge.
	ShiftM, ShiftGeom, ShiftRegions            bool        = true, true, true // should shift act on magnetization, geometry, regions?
)

func init() {
	DeclFunc("Shift", Shift, "Shifts the simulation by +1/-1 cells along X")
	DeclVar("ShiftMagL", &ShiftMagL, "Upon shift, insert this magnetization from the left")
	DeclVar("ShiftMagR", &ShiftMagR, "Upon shift, insert this magnetization from the right")
	DeclVar("ShiftMagU", &ShiftMagU, "Upon shift, insert this magnetization from the top")
	DeclVar("ShiftMagD", &ShiftMagD, "Upon shift, insert this magnetization from the bottom")
	DeclVar("ShiftM", &ShiftM, "Whether Shift() acts on magnetization")
	DeclVar("ShiftGeom", &ShiftGeom, "Whether Shift() acts on geometry")
	DeclVar("ShiftRegions", &ShiftRegions, "Whether Shift() acts on regions")
	DeclVar("TotalShift", &TotalShift, "Amount by which the simulation has been shifted (m).")
}

// position of the window lab frame
func GetShiftPos() float64  { return -TotalShift }
func GetShiftYPos() float64 { return -TotalYShift }

// shift the simulation window over dx cells in X direction
func Shift(dx int) {
	TotalShift += float64(dx) * Mesh().CellSize()[X] // needed to re-init geom, regions
	if ShiftM {
		shiftMag(M.Buffer(), dx) // TODO: M.shift?
		println("trace1")
		shiftKt(M.Buffer(), Kt1, dx)
		shiftKt(M.Buffer(), Kt2, dx)
		shiftKt(M.Buffer(), Kt3, dx)
		println("trace2")
		shiftAnisT(M.Buffer(), AnisT1, dx)
		shiftAnisT(M.Buffer(), AnisT2, dx)
		shiftAnisT(M.Buffer(), AnisT3, dx)
		println("trace3")
	}
	if ShiftRegions {
		regions.shift(dx)
	}
	if ShiftGeom {
		geometry.shift(dx)
	}
	M.normalize()
}

func shiftMag(m *data.Slice, dx int) {
	m2 := cuda.Buffer(1, m.Size())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		cuda.ShiftX(m2, comp, dx, float32(ShiftMagL[c]), float32(ShiftMagR[c]))
		data.Copy(comp, m2) // str0 ?
	}
}

func shiftKt(m *data.Slice, Kt ScalarField, dx int) {
	KBuffer := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(KBuffer)

	// shift field by dx
	comp := ValueOf(Kt.Quantity)
	comp2 := ValueOf(Kt.Quantity)
	defer comp.Free()
	defer comp2.Free()
	newv := float32(1) // initially fill edges with 1's
	cuda.ShiftX(KBuffer, comp, dx, newv, newv)  // fills whole edge by one fixed value
	data.Copy(comp, KBuffer)


	// fill edges with correct values
	n := Mesh().Size()
	x1, x2 := shiftDirtyRange(dx)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
			        if dx > 0 {
					val := cuda.GetCell(comp2, 0, n[X] - dx + ix, iy, iz)
					cuda.SetCell(comp, 0, ix, iy, iz, val)
				} else {
				        val := cuda.GetCell(comp2, 0, - n[X] - dx + ix, iy, iz)
					cuda.SetCell(comp, 0, ix, iy, iz, val)
				}
			}
		}
	}

}

func shiftSingleComp(dst, src *data.Slice, c, dx int, newvalue float32) {
	comp := src.Comp(c)
	// defer comp.Free()
	cuda.ShiftX(dst, comp, dx, newvalue, newvalue)
	data.Copy(comp, dst)
}

func shiftAnisT(m *data.Slice, AnisT VectorField, dx int) {
	KBuffer := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(KBuffer)

	// shift field by dx
	comp := ValueOf(AnisT.Quantity)
	comp2 := ValueOf(AnisT.Quantity)
	defer comp.Free()
	defer comp2.Free()
	newv := float32(1) // initially fill edges with 1's
	// for c := 0; c < 3; c++ {
	//     cuda.ShiftX(KBuffer, AnisT.Comp(c), dx, newv, newv)
	// }
	// data.Copy(comp, KBuffer)
	for c := 0; c < 3; c++ {
            shiftSingleComp(KBuffer, comp, c, dx, newv)
	}


	// fill edges with correct values
	n := Mesh().Size()
	x1, x2 := shiftDirtyRange(dx)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
				for c := 0; c < 3; c++ {
			            if dx > 0 {
				       	val := cuda.GetCell(comp2, c, n[X] - dx + ix, iy, iz)
					cuda.SetCell(comp, c, ix, iy, iz, val)
				    } else {
				        val := cuda.GetCell(comp2, c, - n[X] - dx + ix, iy, iz)
					cuda.SetCell(comp, c, ix, iy, iz, val)
				    }
				}
			}
		}
	}
}

// shift the simulation window over dy cells in Y direction
func YShift(dy int) {
	TotalYShift += float64(dy) * Mesh().CellSize()[Y] // needed to re-init geom, regions
	if ShiftM {
		shiftMagY(M.Buffer(), dy)
	}
	if ShiftRegions {
		regions.shiftY(dy)
	}
	if ShiftGeom {
		geometry.shiftY(dy)
	}
	M.normalize()
}

func shiftMagY(m *data.Slice, dy int) {
	m2 := cuda.Buffer(1, m.Size())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		cuda.ShiftY(m2, comp, dy, float32(ShiftMagU[c]), float32(ShiftMagD[c]))
		data.Copy(comp, m2) // str0 ?
	}
}
