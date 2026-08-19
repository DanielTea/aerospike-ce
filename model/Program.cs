//
// Entry point.
//
//   dotnet run --project model -- spec/demo.json
//
// Opens the PicoGK viewer, prints the derived numbers, writes STLs to out/.
//

using Leap71.AerospikeCEGeometry;
using Leap71.ShapeKernel;
using PicoGK;

namespace AerospikeCE
{
    public static class Program
    {
        private static string _specPath = "spec/demo.json";

        public static void Main(string[] args)
        {
            if (args.Length > 0)
                _specPath = args[0];

            if (!File.Exists(_specPath))
            {
                Console.Error.WriteLine($"spec not found: {_specPath}");
                Environment.Exit(1);
            }

            EngineSpec oSpec = EngineSpec.Load(_specPath);

            try
            {
                Library.Go(oSpec.Geometry.VoxelSizeMm, Task);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine("PicoGK failed to start: " + e.Message);
                Console.Error.WriteLine(
                    "Check that the native runtime is installed and that you are on " +
                    "win-x64 or osx-arm64. See README.");
                Environment.Exit(1);
            }
        }

        private static void Task()
        {
            EngineSpec oSpec = EngineSpec.Load(_specPath);

            PlugContour oContour = PlugContour.Build(
                oSpec.Nozzle.ExpansionRatio,
                oSpec.Nozzle.ExitRadiusMm,
                oSpec.Gas.Gamma,
                oSpec.Nozzle.ContourPoints,
                oSpec.Nozzle.TruncateFraction);

            // ---- derived numbers, always logged so a code agent can read them ----
            Library.Log($"spec              {oSpec.Name}");
            Library.Log($"expansion ratio   {oSpec.Nozzle.ExpansionRatio:F3}");
            Library.Log($"exit Mach         {oContour.ExitMach:F4}");
            Library.Log($"throat angle      {oContour.ThroatAngleRad * 180.0 / Math.PI:F3} deg");
            Library.Log($"throat area       {oContour.ThroatAreaMm2:F3} mm2");
            Library.Log($"spike length      {oContour.LengthMm:F3} mm");
            Library.Log($"tip radius        {oContour.TipRadiusMm:F4} mm");
            Library.Log($"voxel size        {oSpec.Geometry.VoxelSizeMm:F3} mm");

            // ---- geometry ----
            Voxels voxSpike = SpikeGeometry.voxBuildSpike(oContour, oSpec);
            Voxels voxCowl  = SpikeGeometry.voxBuildCowl(oContour, oSpec);
            Voxels voxPlate = SpikeGeometry.voxBuildBasePlate(oContour, oSpec);

            Voxels voxOuter = voxCowl + voxPlate;

            Sh.PreviewVoxels(voxSpike, Cp.clrRock,   0.9f);
            Sh.PreviewVoxels(voxOuter, Cp.clrBlue,   0.5f);

            // ---- invariants a code agent can check without seeing anything ----
            Assert(oContour.R[0] < oContour.ExitRadiusMm,
                   "throat point must sit inboard of the lip");
            Assert(oContour.LengthMm > 0.0,
                   "spike length must be positive");
            if (oSpec.Nozzle.TruncateFraction >= 1.0)
                Assert(oContour.TipRadiusMm < 0.05 * oContour.ExitRadiusMm,
                       "ideal full-length spike must close on the axis");

            BBox3 oBox = voxSpike.oCalculateBoundingBox();
            Library.Log($"spike bbox        {oBox.vecSize()}");

            // ---- export ----
            Directory.CreateDirectory(oSpec.Output.StlDir);
            oContour.WriteCsv(Path.Combine(oSpec.Output.StlDir, "contour_csharp.csv"));

            if (oSpec.Output.ExportSpike)
                Sh.ExportVoxelsToSTLFile(voxSpike,
                    Path.Combine(oSpec.Output.StlDir, $"{oSpec.Name}_spike.stl"));

            if (oSpec.Output.ExportCowl)
                Sh.ExportVoxelsToSTLFile(voxOuter,
                    Path.Combine(oSpec.Output.StlDir, $"{oSpec.Name}_cowl.stl"));

            Library.Log("done");
        }

        private static void Assert(bool bCondition, string strMessage)
        {
            if (!bCondition)
                throw new Exception("INVARIANT VIOLATED: " + strMessage);
            Library.Log($"ok   {strMessage}");
        }
    }
}
