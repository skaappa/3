package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
)

// Preload data to variables:
var (
	kt1fromfile        = LoadFile("mykt1.ovf")
	kt2fromfile        = LoadFile("mykt2.ovf")
	kt3fromfile        = LoadFile("mykt3.ovf")
	anist1fromfile     = LoadFile("myanist1.ovf")
	anist2fromfile     = LoadFile("myanist2.ovf")
	anist3fromfile     = LoadFile("myanist3.ovf")
	ZeroShiftableField = ZeroShiftableFieldLike(kt1fromfile, 1)
)

// Use pre-loaded data to create the objects:
var (
	Kt1    = NewShiftableField(kt1fromfile, 1)
	Kt2    = NewShiftableField(kt2fromfile, 1)
	Kt3    = NewShiftableField(kt3fromfile, 1)
	AnisT1 = NewShiftableField(anist1fromfile, 3)
	AnisT2 = NewShiftableField(anist2fromfile, 3)
	AnisT3 = NewShiftableField(anist3fromfile, 3)
)

// Save data from a ShiftableField object to .ovf file.
func SaveShiftableField(fname string, sfield *ShiftableField) {
	s := sfield.buffer.HostCopy()
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()
	info := data.Meta{Time: 0., Name: "myanis", Unit: "", CellSize: Mesh().CellSize()}
	oommf.WriteOVF2(f, s, info, "text")
}

type ShiftableField struct {
	buffer *data.Slice // data on CPU
	ncomp  int         // Number of components
}

func NewShiftableField(loadedSlice *data.Slice, ncomp int) *ShiftableField {
	return &ShiftableField{loadedSlice, ncomp}
}

func ZeroShiftableFieldLike(loadedSlice *data.Slice, ncomp int) *ShiftableField {
	zeroSlice := data.NewSlice(ncomp, loadedSlice.Size())

	// Make sure it is full zero:
	n := loadedSlice.Size()
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				for comp := 0; comp < ncomp; comp++ {
					zeroSlice.Set(comp, ix, iy, iz, 0.0)
				}
			}
		}
	}
	return &ShiftableField{zeroSlice, ncomp}
}

// Copy data to GPU buffer and return it:
func (s *ShiftableField) Buffer() *data.Slice {
	buffer := cuda.Buffer(s.NComp(), MeshSize())
	data.Copy(buffer, s.buffer)
	return buffer
}

// Shift the underlying field
func (s *ShiftableField) Shift(dx int) {
	dst := cuda.Buffer(s.NComp(), Mesh().Size())
	defer cuda.Recycle(dst)

	gpubuf := s.Buffer()
	defer cuda.Recycle(gpubuf)

	// Shift the field and fill the missing edge grid points with zeros:
	for i := 0; i < s.NComp(); i++ {
		dsti := dst.Comp(i)
		origi := gpubuf.Comp(i)
		if dx != 0 {
			cuda.ShiftX(dsti, origi, dx, 0, 0)
		}
	}

	// Fill in the correct data by copying the leaking data to the missing edge grid points:
	n := Mesh().Size()
	x1, x2 := shiftDirtyRange(dx)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
				for i := 0; i < s.NComp(); i++ {
					dsti := dst.Comp(i)
					if dx > 0 {
						val := cuda.GetCell(gpubuf, i, n[X]-dx+ix, iy, iz)
						cuda.SetCell(dsti, 0, ix, iy, iz, val)
					} else {
						val := cuda.GetCell(gpubuf, i, -n[X]-dx+ix, iy, iz)
						cuda.SetCell(dsti, 0, ix, iy, iz, val)
					}
				}
			}
		}
	}
	data.Copy(s.buffer, dst)
}

func (s *ShiftableField) NComp() int { return s.ncomp }
func (s *ShiftableField) Free()      { s.buffer.Free() }
