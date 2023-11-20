package engine

import (
	"fmt"
	"github.com/mumax/3/data"
)

var (
	DWPos   = NewScalarValue("ext_dwpos", "m", "Position of the simulation window while following a domain wall", GetShiftPos) // TODO: make more accurate
	DWxPos  = NewScalarValue("ext_dwxpos", "m", "Position of the simulation window while following a domain wall", GetDWxPos)
	// DWSpeed = NewScalarValue("ext_dwspeed", "m/s", "Speed of the simulation window while following a domain wall", getShiftSpeed)
	DWSpeed = NewScalarValue("ext_dwspeed", "m/s", "Speed of the domain wall", getDWSpeed)
)

func init() {
	DeclFunc("ext_centerWall", CenterWall, "centerWall(c) shifts m after each step to keep m_c close to zero")
	DeclFunc("centerWall", centerWall, "samis empty doc")
}

func centerWall(c int) {
	M := &M
	mc := sAverageUniverse(M.Buffer().Comp(c))[0]
	n := Mesh().Size()
	tolerance := 1 / float64(n[X]) // x*2 * expected <m> change for 1 cell shift

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero {
		sign := magsign(M.GetCell(0, n[Y]/2, n[Z]/2)[c])
		ShiftMagL[c] = float64(sign)
		ShiftMagR[c] = -float64(sign)
	}

	sign := magsign(ShiftMagL[c])

	//log.Println("mc", mc, "tol", tolerance)

	if mc < -tolerance {
		Shift(sign)
	} else if mc > tolerance {
		Shift(-sign)
	}
}

// This post-step function centers the simulation window on a domain wall
// between up-down (or down-up) domains (like in perpendicular media). E.g.:
// 	PostStep(CenterPMAWall)
func CenterWall(magComp int) {
	PostStep(func() { centerWall(magComp) })
}

func magsign(x float64) int {
	if x > 0.1 {
		return 1
	}
	if x < -0.1 {
		return -1
	}
	panic(fmt.Errorf("center wall: unclear in which direction to shift: magnetization at border=%v. Set ShiftMagL, ShiftMagR", x))
}

// used for speed
var (
	lastShift float64 // shift the last time we queried speed
	lastT     float64 // time the last time we queried speed
	lastV     float64 // speed the last time we queried speed
	lastDWxPos float64  // Sami
	lastSpeed float64  // Sami
)

func getShiftSpeed() float64 {
	if lastShift != GetShiftPos() {
		lastV = (GetShiftPos() - lastShift) / (Time - lastT)
		lastShift = GetShiftPos()
		lastT = Time
	}
	return lastV
}

func getDWSpeed() float64 {  // Samis function
        pos := GetDWxPos()

	if lastT == Time {
	    lastT = Time
	    return lastSpeed
	}

	speed := (pos - lastDWxPos) / (Time - lastT)
	if lastDWxPos != pos {
		lastDWxPos = pos
		lastT = Time
	}
	lastSpeed = speed
	return speed
}

func GetDWxPos() float64 {
	M := &M
	// mx := sAverageUniverse(M.Buffer().Comp(0))[0]
	my := sAverageUniverse(M.Buffer().Comp(1))[0]
	c := Mesh().CellSize()
	n := Mesh().Size()
	position := my * c[0] * float64(n[0]) / 2.

	// position := SamisDWxPos()
        
	return GetShiftPos() + position
}

// func SamisDWxPos_slow() float64 {
// 	// Position is the average x-value where the y-component
// 	// turns negative (averaged over the y-axis):

// 	M := &M
// 	c := Mesh().CellSize()
// 	n := Mesh().Size()

// 	avg := make([]float64, n[Y])
// 	for y := 0; y < n[Y]; y++ {
// 		    for x := 1; x < n[X]; x++ {
// 		        if M.GetCell(x, y, 0)[1] < 0.0 {
//      	                    avg[y] = float64(x)
// 			    break
// 		        }
// 		    }
// 	}
// 	position := Sum(avg) / float64(n[Y]) * c[0]
// 	return position
// }

