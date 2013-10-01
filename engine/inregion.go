package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// constrains Slicer to a region
type inRegion struct {
	q      Slicer
	region int
}

func (q *inRegion) NComp() int                 { return q.q.NComp() }
func (q *inRegion) Name() string               { return fmt.Sprint(q.q.Name(), ".region", q.region) }
func (q *inRegion) Unit() string               { return q.q.Unit() }
func (q *inRegion) Mesh() *data.Mesh           { return q.q.Mesh() }
func (q *inRegion) Slice() (*data.Slice, bool) { return getRegion(q.q, q.region) }
func (q *inRegion) TableData() []float64       { return Average(q) }
func (q *inRegion) volume() float64            { return regions.volume(q.region) }

// constrains inputParam to a region
type selectRegion struct {
	q      *inputParam
	region int
}

func (p *selectRegion) TableData() []float64       { return p.q.getRegion(p.region) }
func (p *selectRegion) NComp() int                 { return p.q.NComp() }
func (p *selectRegion) Name() string               { return fmt.Sprint(p.q.Name(), ".region", p.region) }
func (p *selectRegion) Unit() string               { return p.q.Unit() }
func (p *selectRegion) Slice() (*data.Slice, bool) { return getRegion(p.q, p.region) }
func (p *selectRegion) Mesh() *data.Mesh           { return p.q.Mesh() }
func (p *selectRegion) volume() float64            { return regions.volume(p.region) }

func getRegion(q Slicer, region int) (*data.Slice, bool) {
	src, r := q.Slice()
	if r {
		defer cuda.Recycle(src)
	}
	out := cuda.Buffer(q.NComp(), q.Mesh())
	cuda.RegionSelect(out, src, regions.Gpu(), byte(region))
	return out, true
}