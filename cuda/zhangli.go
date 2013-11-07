package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

func AddZhangLiTorque(torque, m, J *data.Slice, bsat, alpha, xi, pol LUTPtr, regions *Bytes) {

	mesh := torque.Mesh()

	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	k_addzhanglitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		J.DevPtr(X), J.DevPtr(Y), J.DevPtr(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		unsafe.Pointer(bsat), unsafe.Pointer(alpha), unsafe.Pointer(xi), unsafe.Pointer(pol),
		regions.Ptr, N[X], N[Y], N[Z], mesh.PBC_code(), cfg, stream0)
}
