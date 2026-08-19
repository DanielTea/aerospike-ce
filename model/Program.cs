//
// Entry point.
//
//   dotnet run --project model -- spec/demo.json
//
// Opens the PicoGK viewer, prints the derived numbers, writes STLs to out/.
//
// Every derived quantity is logged. That log is the only way a future agent,
// which cannot see the viewer, can tell what this run actually produced -- and
// it is what gets diffed against the Python reference output.
//

using System.Globalization;

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
            // The whole point of the log is that its numbers get diffed against
            // the Python reference. On a machine with a comma decimal separator
            // that diff turns into noise, and contour_csharp.csv becomes a file
            // where the decimal point and the field delimiter are both a comma.
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

            if (args.Length > 0)
                _specPath = args[0];

            if (!File.Exists(_specPath))
            {
                Console.Error.WriteLine($"spec not found: {_specPath}");
                Environment.Exit(1);
            }

            try
            {
                EngineSpec oSpec = EngineSpec.Load(_specPath);
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

            EngineAssembly oAsm = EngineAssembly.Build(oContour, oSpec);

            // ---- derived numbers, logged so they can be diffed against Python ----
            Library.Log($"spec               {oSpec.Name}");
            Library.Log($"expansion ratio    {oSpec.Nozzle.ExpansionRatio:F4}");
            Library.Log($"exit Mach          {oContour.ExitMach:F6}");
            Library.Log($"throat angle       {oContour.ThroatAngleRad * 180.0 / Math.PI:F6} deg");
            Library.Log($"throat area        {oContour.ThroatAreaMm2:F6} mm2");
            Library.Log($"throat area (geom) {oContour.ThroatAreaFromGeometryMm2:F6} mm2");
            Library.Log($"lip station        {oContour.LipXMm:F6} mm");
            Library.Log($"throat slant       {oContour.ThroatSlantLengthMm:F6} mm");
            Library.Log($"spike length       {oContour.LengthMm:F6} mm");
            Library.Log($"tip radius         {oContour.TipRadiusMm:F6} mm");
            Library.Log($"shoulder radius    {oAsm.ShoulderRadiusMm:F6} mm");
            Library.Log($"chamber outer R    {oAsm.ChamberOuterRadiusMm:F6} mm");
            Library.Log($"chamber area       {oAsm.ChamberAreaMm2:F6} mm2");
            Library.Log($"contraction ratio  {oAsm.ChamberAreaMm2 / oContour.ThroatAreaMm2:F6}");
            Library.Log($"bore radius        {oAsm.BoreRadiusMm:F6} mm");
            Library.Log($"flange radius      {oAsm.FlangeRadiusMm:F6} mm");
            Library.Log($"head station       {oAsm.HeadXMm:F6} mm");
            Library.Log($"overall length     {oAsm.OverallLengthMm:F6} mm");
            Library.Log($"overall diameter   {2.0 * oAsm.OverallRadiusMm:F6} mm");
            Library.Log($"cowl wall ends at  x={oAsm.OuterWallX[^1]:F6} r={oAsm.OuterWallR[^1]:F6}");
            Library.Log($"voxel size         {oSpec.Geometry.VoxelSizeMm:F4} mm");

            // ---- invariants a code agent can check without seeing anything ----

            // Upstream end of the contour: the lip placement is what makes the
            // inclined throat area come out right.
            AssertClose(oContour.ThroatAreaFromGeometryMm2, oContour.ThroatAreaMm2, 1e-9,
                        "inclined throat area reproduces pi*r_e^2/eps");
            AssertClose(oContour.ThroatInclinationRad + oContour.ThroatAngleRad,
                        Math.PI / 2.0, 1e-9,
                        "sonic line sits at pi/2 - nu_e from the axis");

            // The load-bearing one: the cowl wall is solved from an area
            // schedule that knows only A_t and nu_e, yet must land on the lip.
            AssertClose(oAsm.OuterWallX[^1], oContour.LipXMm, 1e-6,
                        "contraction lands on the lip in x");
            AssertClose(oAsm.OuterWallR[^1], oContour.ExitRadiusMm, 1e-6,
                        "contraction lands on the lip in r");

            // Downstream end: the ideal full spike closes on the axis.
            if (oSpec.Nozzle.TruncateFraction >= 1.0)
                Assert(oContour.TipRadiusMm < 0.05 * oContour.ExitRadiusMm,
                       "ideal full-length spike closes on the axis");

            // The duct must choke at the throat and nowhere else.
            var (fMinX, fMinArea) = oAsm.MinimumFlowArea();
            Library.Log($"min flow area      {fMinArea:F6} mm2 at x={fMinX:F6} mm");
            AssertClose(fMinArea / oContour.ThroatAreaMm2, 1.0, 2e-3,
                        "minimum duct area equals the throat area");
            AssertClose(fMinX, oContour.LipXMm, 0.05,
                        "the choke point is at the throat, not upstream of it");

            // ---- geometry ----
            Voxels voxCentrebody = SpikeGeometry.voxBuildCentrebody(oAsm);
            Voxels voxCowl       = SpikeGeometry.voxBuildCowl(oAsm);
            Voxels voxHead       = SpikeGeometry.voxBuildHead(oAsm);

            Sh.PreviewVoxels(voxCentrebody, Cp.clrRock, 0.9f);
            Sh.PreviewVoxels(voxCowl,       Cp.clrBlue, 0.5f);
            Sh.PreviewVoxels(voxHead,       Cp.clrRock, 0.7f);

            LogBox("centrebody", voxCentrebody);
            LogBox("cowl",       voxCowl);
            LogBox("head",       voxHead);

            // ---- export ----
            Directory.CreateDirectory(oSpec.Output.StlDir);
            oContour.WriteCsv(Path.Combine(oSpec.Output.StlDir, "contour_csharp.csv"));

            if (oSpec.Output.ExportSpike)
                Sh.ExportVoxelsToSTLFile(voxCentrebody,
                    Path.Combine(oSpec.Output.StlDir, $"{oSpec.Name}_centrebody.stl"));

            if (oSpec.Output.ExportCowl)
                Sh.ExportVoxelsToSTLFile(voxCowl,
                    Path.Combine(oSpec.Output.StlDir, $"{oSpec.Name}_cowl.stl"));

            if (oSpec.Output.ExportHead)
                Sh.ExportVoxelsToSTLFile(voxHead,
                    Path.Combine(oSpec.Output.StlDir, $"{oSpec.Name}_head.stl"));

            Library.Log("done");
        }

        private static void LogBox(string strName, Voxels vox)
        {
            BBox3 oBox = vox.oCalculateBoundingBox();
            Library.Log($"bbox {strName,-12} {oBox.vecSize()}");
        }

        private static void Assert(bool bCondition, string strMessage)
        {
            if (!bCondition)
                throw new Exception("INVARIANT VIOLATED: " + strMessage);
            Library.Log($"ok   {strMessage}");
        }

        private static void AssertClose(double fGot, double fWant, double fTol, string strMessage)
        {
            double fErr = Math.Abs(fGot - fWant);
            if (fErr > fTol)
                throw new Exception(
                    $"INVARIANT VIOLATED: {strMessage} (got {fGot:G17}, want {fWant:G17}, " +
                    $"err {fErr:G6} > tol {fTol:G6})");
            Library.Log($"ok   {strMessage}  (err {fErr:G3})");
        }
    }
}
