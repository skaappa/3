package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type ShiftableField struct {
        quantity   Quantity
        buffer     *data.Slice
}

func (s ShiftableField) xinit() {  // This is called in mesh.go right after initing mesh
	q := ValueOf(s.quantity)
	defer cuda.Recycle(q)

	buffer := cuda.NewSlice(s.NComp(), MeshSize())
	// defer buffer.Recycle()

	data.Copy(buffer, q)
	s.buffer = cuda.NewSlice(s.NComp(), MeshSize())
	data.Copy(s.buffer, buffer)

	println("Here xinit")
	println(s.buffer)
}

func (s ShiftableField) resize() {
        println("Here resize")
	println(s.buffer)
	s.xinit()
	backup := s.Buffer().HostCopy()
	s2 := Mesh().Size()
	resized := data.Resample(backup, s2)
	s.Buffer().Free()
	s.buffer = cuda.NewSlice(s.NComp(), s2)
	data.Copy(s.buffer, resized)
}



func NewShiftableField(name, unit, desc string, ncomp int, f func(dst *data.Slice)) ShiftableField {
	q := AsShiftableField(&fieldFunc{info{ncomp, name, unit}, f})
	DeclROnly(name, q, cat(desc, unit))
	return q
}

func AsShiftableField(q Quantity) ShiftableField {
	// println("Here AsShiftableField")
        return ShiftableField{q, nil}
}

// func (s ShiftableField) alloc() {
//         buffer := cuda.NewSlice(s.NComp(), MeshSize())
//         s.buffer = buffer
// }

func (s ShiftableField) Buffer() *data.Slice {
        return s.buffer
}

func (s ShiftableField) Shift(dx int) {
	dst := cuda.Buffer(s.NComp(), Mesh().Size())
	defer cuda.Recycle(dst)

        for i := 0; i < s.NComp(); i++ {
                dsti := dst.Comp(i)
                origi := s.buffer.Comp(i)
                if dx != 0 {
                        cuda.ShiftX(dsti, origi, dx, 0, 0)
                }
        }
        n := Mesh().Size()
        x1, x2 := shiftDirtyRange(dx)

        for iz := 0; iz < n[Z]; iz++ {
                for iy := 0; iy < n[Y]; iy++ {
                        for ix := x1; ix < x2; ix++ {
                                for i := 0; i < s.buffer.NComp(); i++ {
                                        dsti := dst.Comp(i)
                                        if dx > 0 {
                                                val := cuda.GetCell(s.buffer, i, n[X]-dx+ix, iy, iz)
                                                cuda.SetCell(dsti, 0, ix, iy, iz, val)
                                        } else {
                                                val := cuda.GetCell(s.buffer, i, -n[X]-dx+ix, iy, iz)
                                                cuda.SetCell(dsti, 0, ix, iy, iz, val)
                                        }
                                }
                        }
                }
        }
	data.Copy(s.buffer, dst)
        println("SHIFTED")
}

func (s ShiftableField) NComp() int { return s.quantity.NComp() }
func (s ShiftableField) Free() { s.buffer.Free() }