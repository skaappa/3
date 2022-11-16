package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"

)

var (
    kt1fromfile = LoadFile("mykt1.ovf")
    kt2fromfile = LoadFile("mykt2.ovf")
    kt3fromfile = LoadFile("mykt3.ovf")
    anist1fromfile = LoadFile("myanist1.ovf")
    anist2fromfile = LoadFile("myanist2.ovf")
    anist3fromfile = LoadFile("myanist3.ovf")
    sZeroScalar = NewShiftableField(kt1fromfile, 1)
)

var (
	Kt1        = NewShiftableField(kt1fromfile, 1)
	Kt2        = NewShiftableField(kt2fromfile, 1)
	Kt3        = NewShiftableField(kt3fromfile, 1)
	AnisT1     = NewShiftableField(anist1fromfile, 3)
	AnisT2     = NewShiftableField(anist2fromfile, 3)
	AnisT3     = NewShiftableField(anist3fromfile, 3)
)



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

func mysave(fname string, sfield *ShiftableField) {
     s := sfield.buffer.HostCopy()
     f, err := httpfs.Create(fname)
     util.FatalErr(err)
     defer f.Close()
     info := data.Meta{Time: 0., Name: "myanis", Unit: "", CellSize: Mesh().CellSize()}
     oommf.WriteOVF2(f, s, info, "text")
}


type ShiftableField struct {
        buffer     *data.Slice
	ncomp      int
}

func NewShiftableField(loadedSlice *data.Slice, ncomp int) *ShiftableField {
        return &ShiftableField{loadedSlice, ncomp}
}

func (s *ShiftableField) Buffer() *data.Slice {
        buffer := cuda.Buffer(s.NComp(), MeshSize())
	data.Copy(buffer, s.buffer)
        return buffer
}

func (s *ShiftableField) Shift(dx int) {
	dst := cuda.Buffer(s.NComp(), Mesh().Size())
	defer cuda.Recycle(dst)

	gpubuf := s.Buffer()
	defer cuda.Recycle(gpubuf)

        for i := 0; i < s.NComp(); i++ {
                dsti := dst.Comp(i)
                origi := gpubuf.Comp(i)
                if dx != 0 {
                        cuda.ShiftX(dsti, origi, dx, 0, 0)
                }
        }
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
func (s *ShiftableField) Free() { s.buffer.Free() }