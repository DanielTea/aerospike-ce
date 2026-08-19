//
// Turns the contour into voxel geometry using PicoGK + LEAP 71 ShapeKernel.
//
// Note what is NOT here: no sketches, no extrudes, no feature tree. The spike is
// a surface of revolution whose radius is a *function*, evaluated by the kernel
// at whatever resolution the voxel field demands. That function-first mindset is
// the whole point of the computational engineering approach.
//

using System.Numerics;
using PicoGK;

namespace Leap71
{
    using ShapeKernel;

    namespace AerospikeCEGeometry
    {
        using global::AerospikeCE;

        public static class SpikeGeometry
        {
            /// <summary>The plug / centrebody, as a solid of revolution.</summary>
            public static Voxels voxBuildSpike(PlugContour oContour, EngineSpec oSpec)
            {
                LocalFrame oFrame = new LocalFrame(new Vector3(0, 0, 0), Vector3.UnitZ);

                BaseCylinder oSpike = new BaseCylinder(
                    oFrame,
                    (float)oContour.LengthMm,
                    (float)oContour.ExitRadiusMm);

                // radius as a function of normalised station along the axis
                oSpike.SetLengthSteps(600);
                oSpike.SetRadius(new SurfaceModulation(
                    new LineModulation(fRatio => (float)oContour.RadiusAtLengthRatio(fRatio))));

                return oSpike.voxConstruct();
            }

            /// <summary>
            /// The outer cowl ring that forms the other side of the annular throat.
            /// Deliberately simple: a straight ring at the lip radius. The real
            /// engineering, injector, manifold, cooling, is not modelled here.
            /// </summary>
            public static Voxels voxBuildCowl(PlugContour oContour, EngineSpec oSpec)
            {
                float fLipRadius = (float)oContour.ExitRadiusMm;
                float fThickness = oSpec.Geometry.CowlThicknessMm;
                float fLength    = oSpec.Geometry.CowlLengthMm;

                // sits upstream of x = 0 so the lip lands on the throat station
                LocalFrame oFrame = new LocalFrame(new Vector3(0, 0, -fLength), Vector3.UnitZ);

                BasePipe oCowl = new BasePipe(
                    oFrame,
                    fLength,
                    fLipRadius,
                    fLipRadius + fThickness);

                oCowl.SetLengthSteps(120);
                return oCowl.voxConstruct();
            }

            /// <summary>
            /// A flat mounting plate behind the throat so the demonstrator has
            /// something to bolt to and a flat face to print from.
            /// </summary>
            public static Voxels voxBuildBasePlate(PlugContour oContour, EngineSpec oSpec)
            {
                float fThickness = oSpec.Geometry.BasePlateThicknessMm;
                float fRadius    = (float)oContour.ExitRadiusMm + oSpec.Geometry.CowlThicknessMm;

                LocalFrame oFrame = new LocalFrame(
                    new Vector3(0, 0, -oSpec.Geometry.CowlLengthMm - fThickness),
                    Vector3.UnitZ);

                BaseCylinder oPlate = new BaseCylinder(oFrame, fThickness, fRadius);
                return oPlate.voxConstruct();
            }
        }
    }
}
