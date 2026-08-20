//
// Voxelisation.
//
// Hands a bounded implicit to PicoGK and gets voxels back. That is the entire
// contents of this file, and deliberately so: the distance functions live in
// CooledGeometry.cs, and the dimensions they use were decided in validate/.
//
// The revolve-and-boolean builders that used to live here are gone with the
// mirrored physics. They took an EngineAssembly and rebuilt each part from a
// contour the C# had derived for itself; there is no longer a C# contour to
// derive from, and no reason for there to be one.
//

using PicoGK;

namespace Leap71
{
    using ShapeKernel;

    namespace AerospikeCEGeometry
    {
        using global::AerospikeCE;

        public static class SpikeGeometry
        {
            /// <summary>
            /// A part built from its signed distance function, with cooling
            /// channels, feed ports, manifolds, mounting lugs and injector
            /// orifices already in it.
            ///
            /// PicoGK renders any bounded implicit straight into voxels, so the
            /// same formulation the Python mesher uses goes to the kernel
            /// unchanged. The alternative -- booleaning several hundred
            /// individual channel solids -- would be slower and would need a
            /// second description of the same geometry to keep in step.
            ///
            /// Voxel size has to resolve the narrowest feature. Budget three
            /// voxels across it; Program.cs warns when the plan does not.
            /// </summary>
            public static Voxels voxBuildCooledPart(CooledPart oPart)
            {
                return new Voxels(oPart);
            }
        }
    }
}
