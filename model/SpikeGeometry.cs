//
// Turns the assembly profiles into voxel geometry using PicoGK + LEAP 71
// ShapeKernel.
//
// Note what is NOT here: no sketches, no extrudes, no feature tree, and no
// physics. Every part is a surface of revolution whose radius is a *function*,
// evaluated by the kernel at whatever resolution the voxel field demands. The
// functions themselves come from EngineAssembly; this file only sweeps them.
//
// Model x maps to PicoGK Z. Lengths stay in mm throughout.
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
            private const uint nLengthSteps = 600;

            /// <summary>
            /// Solid of revolution between two axial stations, radius given as a
            /// function of x.
            ///
            /// SetRadius resets the length steps internally, so it has to be
            /// called before SetLengthSteps and not after. Called the other way
            /// round the sweep quietly drops to the 500-step default and the
            /// shoulder loses definition.
            /// </summary>
            private static Voxels voxRevolve(
                double fX0, double fX1, Func<double, double> fnRadiusAtX)
            {
                float fLength = (float)(fX1 - fX0);
                LocalFrame oFrame = new LocalFrame(
                    new Vector3(0, 0, (float)fX0), Vector3.UnitZ);

                BaseCylinder oShape = new BaseCylinder(oFrame, fLength, 1f);
                oShape.SetRadius(new SurfaceModulation(new LineModulation(
                    fRatio => (float)fnRadiusAtX(fX0 + fRatio * (fX1 - fX0)))));
                oShape.SetLengthSteps(nLengthSteps);
                return oShape.voxConstruct();
            }

            /// <summary>
            /// The centrebody: cylindrical through the chamber, then the
            /// Angelino spike, hollowed and capped behind the truncation.
            ///
            /// Built as solid minus cavity rather than as a pipe. The voxel
            /// erosion the kernel does here is a true distance-field offset, so
            /// unlike the polyline offset in the Python reference it cannot
            /// fold over at the shoulder and needs no pruning. The two agree on
            /// the flow surface exactly and differ by a fraction of a
            /// millimetre inside the shoulder, where the Python is the more
            /// conservative of the two.
            /// </summary>
            public static Voxels voxBuildCentrebody(EngineAssembly oAsm)
            {
                Voxels voxSolid = voxRevolve(
                    oAsm.HeadXMm, oAsm.TipXMm, oAsm.CentrebodyRadiusAtX);

                // start the cavity ahead of the head face so the bore cuts
                // clean through it and powder has somewhere to go
                Voxels voxCavity = voxRevolve(
                    oAsm.HeadXMm - 1.0, oAsm.CavityEndXMm, oAsm.CavityRadiusAtX);

                return voxSolid - voxCavity;
            }

            /// <summary>
            /// The cowl: outer side of the annular chamber and of the throat,
            /// ending at the lip. Downstream of the lip an aerospike has no
            /// outer wall, which is the entire point of the architecture.
            /// </summary>
            public static Voxels voxBuildCowl(EngineAssembly oAsm)
            {
                double fX0 = oAsm.HeadXMm;
                double fX1 = oAsm.Contour.LipXMm;
                float fLength = (float)(fX1 - fX0);

                LocalFrame oFrame = new LocalFrame(
                    new Vector3(0, 0, (float)fX0), Vector3.UnitZ);

                BasePipe oPipe = new BasePipe(oFrame, fLength, 1f, 2f);
                oPipe.SetRadius(
                    new SurfaceModulation(new LineModulation(
                        fRatio => (float)oAsm.CowlInnerRadiusAtX(fX0 + fRatio * (fX1 - fX0)))),
                    new SurfaceModulation(new LineModulation(
                        fRatio => (float)oAsm.CowlOuterRadiusAtX(fX0 + fRatio * (fX1 - fX0)))));
                oPipe.SetLengthSteps(nLengthSteps);
                return oPipe.voxConstruct();
            }

            /// <summary>
            /// A part built from its signed distance function, with cooling
            /// channels and injector orifices already subtracted.
            ///
            /// PicoGK renders any bounded implicit straight into voxels, so the
            /// distance functions in CooledGeometry.cs -- the same ones the
            /// Python mesher uses -- go to the kernel unchanged. The
            /// alternative, booleaning several hundred individual channel
            /// solids, would be slower and would need a second formulation of
            /// the same geometry to keep in step with the first.
            ///
            /// Voxel size has to resolve the channel. A 0.4 mm channel at a
            /// 0.4 mm voxel is one voxel wide and will come out broken; budget
            /// three voxels across the narrowest feature.
            /// </summary>
            public static Voxels voxBuildCooledPart(CooledPart oPart)
            {
                return new Voxels(oPart);
            }

            /// <summary>
            /// The head closure and mounting flange: one annular disc that caps
            /// the chamber, butts against both the cowl and the centrebody, and
            /// gives the demonstrator a flat face to print from and bolt to.
            /// It is an annulus rather than a plate so the centrebody bore
            /// stays open.
            ///
            /// Deliberately blank. This is not an injector: no orifice pattern,
            /// no manifolding, no propellant routing. See CLAUDE.md.
            /// </summary>
            public static Voxels voxBuildHead(EngineAssembly oAsm)
            {
                float fThickness = (float)oAsm.Spec.Geometry.HeadThicknessMm;
                LocalFrame oFrame = new LocalFrame(
                    new Vector3(0, 0, (float)(oAsm.HeadXMm - fThickness)), Vector3.UnitZ);

                BasePipe oDisc = new BasePipe(
                    oFrame,
                    fThickness,
                    (float)oAsm.BoreRadiusMm,
                    (float)oAsm.FlangeRadiusMm);

                oDisc.SetLengthSteps(60);
                return oDisc.voxConstruct();
            }
        }
    }
}
