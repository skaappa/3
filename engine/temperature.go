package engine

import (
	"github.com/barnex/cuda5/curand"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

var (
	Temp        ScalarParam
	temp_red    derivedParam
	E_therm     *GetScalar
	Edens_therm sAdder
	B_therm     thermField
)

// thermField calculates and caches thermal noise.
type thermField struct {
	seed      int64 // seed for generator
	generator curand.Generator
	noise     *data.Slice // noise buffer
	step      int         // solver step corresponding to noise
	dt        float64     // solver timestep corresponding to noise
}

func init() {
	Temp.init("Temp", "K", "Temperature", []derived{&temp_red})
	DeclFunc("ThermSeed", ThermSeed, "Set a random seed for thermal noise")
	E_therm = NewGetScalar("E_therm", "J", "Thermal energy", GetThermalEnergy)
	Edens_therm.init("Edens_therm", "J/m3", "Thermal energy density", addEdens(&B_therm, -1))
	registerEnergy(GetThermalEnergy, Edens_therm.AddTo)
	B_therm.step = -1 // invalidate noise cache
	DeclROnly("B_therm", &B_therm, "Thermal field (T)")

	// reduced temperature = (alpha * T) / (mu0 * Msat)
	temp_red.init(1, []updater{&Alpha, &Temp, &Msat}, func(p *derivedParam) {
		dst := temp_red.cpu_buf
		alpha := Alpha.cpuLUT()
		T := Temp.cpuLUT()
		Ms := Msat.cpuLUT()
		for i := 0; i < NREGION; i++ { // not regions.MaxReg!
			dst[0][i] = safediv(alpha[0][i]*T[0][i], mag.Mu0*Ms[0][i])
		}
	})
}

func (b *thermField) AddTo(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		cuda.Madd2(dst, dst, b.noise, 1, 1)
	}
}

func (b *thermField) update() {
	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
	}

	util.AssertMsg(Solver.FixDt != 0, "Temperature requires fixed time step")

	// keep constant during time step
	if Solver.NSteps == b.step && Solver.Dt_si == b.dt {
		return
	}

	cuda.Memset(b.noise, 0, 0, 0)
	N := Mesh().NCell()
	kmu0_VgammaDt := mag.Mu0 * mag.Kb / (GammaLL * cellVolume() * Solver.Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.AddTemperature(dst.Comp(i), noise, temp_red.gpuLUT1(), kmu0_VgammaDt, regions.Gpu())
	}

	b.step = Solver.NSteps
	b.dt = Solver.Dt_si
}

func GetThermalEnergy() float64 {
	return -cellVolume() * dot(&M_full, &B_therm)
}

// Seeds the thermal noise generator
func ThermSeed(seed int) {
	B_therm.seed = int64(seed)
	if B_therm.generator != 0 {
		B_therm.generator.SetSeed(B_therm.seed)
	}
}

func (b *thermField) Mesh() *data.Mesh   { return Mesh() }
func (b *thermField) NComp() int         { return 3 }
func (b *thermField) Name() string       { return "Thermal field" }
func (b *thermField) Unit() string       { return "T" }
func (b *thermField) average() []float64 { return qAverageUniverse(b) }
func (b *thermField) Slice() (*data.Slice, bool) {
	b.update()
	return b.noise, false
}
