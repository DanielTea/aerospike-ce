//
// Entry point.
//
//   python validate/build_plan.py --spec spec/regen.json --out out/plan.json
//   dotnet run --project model -- out/plan.json
//
// Reads a build plan and voxelises it. That is the whole job. Every number in
// the plan was decided in validate/, which owns the physics and the sizing;
// this side owns the kernel and nothing else.
//
// The split replaced a mirror -- two implementations of the same maths kept in
// step by hand, with divergence named in CLAUDE.md as the failure mode the
// structure existed to prevent. The mirror never caught an error in the Python;
// every bug the pairing found was a bug in the copy.
//

using System.Globalization;

using Leap71.AerospikeCEGeometry;
using Leap71.ShapeKernel;
using PicoGK;

namespace AerospikeCE
{
    public static class Program
    {
        private static string _planPath = "out/plan.json";
        private static bool _bExitWhenDone;
        private static BuildPlan _oPlan = null!;

        public static void Main(string[] args)
        {
            // The log's numbers get compared against the Python that produced
            // them. Under a comma decimal separator that comparison is noise.
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

            _bExitWhenDone = args.Any(a => a == "--exit-when-done");
            string[] positional = args.Where(a => !a.StartsWith("--")).ToArray();
            if (positional.Length > 0)
                _planPath = positional[0];

            if (!File.Exists(_planPath))
            {
                Console.Error.WriteLine($"build plan not found: {_planPath}");
                Console.Error.WriteLine(
                    "Generate one first:  python validate/build_plan.py --spec spec/regen.json");
                Environment.Exit(1);
            }

            try
            {
                _oPlan = BuildPlan.Load(_planPath);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine("could not read the build plan: " + e.Message);
                Environment.Exit(1);
            }

            try
            {
                Library.Go((float)_oPlan.VoxelSizeMm, Task, bEndAppWithTask: _bExitWhenDone);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine("PicoGK failed to start: " + e.Message);
                Console.Error.WriteLine(
                    "Check that the native runtime is present under vendor/PicoGK/native " +
                    "and that you are on win-x64 or osx-arm64. See README.");
                Environment.Exit(1);
            }
        }

        private static void Task()
        {
            BuildPlan plan = _oPlan;

            Library.Log($"plan               {plan.Name}  (version {plan.PlanVersion})");
            Library.Log($"thrust             {plan.Summary.ThrustSeaLevelN:F1} N sl, "
                        + $"{plan.Summary.ThrustVacuumN:F1} N vac");
            Library.Log($"Isp                {plan.Summary.IspSeaLevelS:F2} s sl, "
                        + $"{plan.Summary.IspVacuumS:F2} s vac");
            Library.Log($"chamber pressure   {plan.Summary.ChamberPressureBar:F2} bar");
            Library.Log($"overall            {plan.Summary.OverallLengthMm:F1} x "
                        + $"{plan.Summary.OverallDiameterMm:F1} mm");
            Library.Log($"voxel size         {plan.VoxelSizeMm:F3} mm");

            CheckResolution(plan);

            // A quarter of a voxel keeps the tabulated profile's interpolation
            // error far below the sampling the kernel is about to do.
            double fStep = plan.VoxelSizeMm / 4.0;

            var oVox = new Dictionary<string, Voxels>();
            foreach (string part in plan.Parts.Keys)
            {
                DateTime t0 = DateTime.Now;
                oVox[part] = SpikeGeometry.voxBuildCooledPart(plan.PartImplicit(part, fStep));
                Library.Log($"built {part,-12} in {(DateTime.Now - t0).TotalSeconds:F1} s   "
                            + $"bbox {oVox[part].oCalculateBoundingBox().vecSize()}");
            }

            Sh.PreviewVoxels(oVox["centrebody"], Cp.clrRock, 0.9f);
            Sh.PreviewVoxels(oVox["cowl"], Cp.clrBlue, 0.5f);
            if (oVox.ContainsKey("head"))
                Sh.PreviewVoxels(oVox["head"], Cp.clrRock, 0.7f);

            string strDir = Path.GetDirectoryName(Path.GetFullPath(_planPath)) ?? "out";
            Directory.CreateDirectory(strDir);
            foreach (var (part, vox) in oVox)
            {
                string strPath = Path.Combine(strDir, $"{plan.Name}_{part}.stl");
                Sh.ExportVoxelsToSTLFile(vox, strPath);
                Library.Log($"wrote {strPath}");
            }
            Library.Log("done");
        }

        /// <summary>
        /// Warn when the voxel size cannot resolve the finest thing in the plan.
        ///
        /// Marching cubes needs roughly three samples across a feature. Coarser,
        /// and the cooling channels come out broken -- which is invisible,
        /// because a part with its channels missing still meshes, still closes,
        /// and still looks like an engine.
        /// </summary>
        private static void CheckResolution(BuildPlan plan)
        {
            double fNarrow = plan.NarrowestFeatureMm();
            if (fNarrow <= 0.0)
                return;
            double fAcross = fNarrow / plan.VoxelSizeMm;
            Library.Log($"narrowest feature  {fNarrow:F3} mm = {fAcross:F1} voxels across");
            if (fAcross < 3.0)
                Library.Log($"WARNING at {plan.VoxelSizeMm:F3} mm the finest features are under "
                            + $"three voxels wide and will render broken. Drop voxel_size_mm to "
                            + $"about {fNarrow / 3.0:F3} mm, remembering memory goes as the cube.");
        }
    }
}
