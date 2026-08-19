//
// Entry point.
//
//   dotnet run --project model -- spec/regen.json
//
// Runs the whole model -- contour, assembly, chamber, injector, cooling -- then
// builds the voxel geometry, opens the viewer and writes STLs.
//
// Every derived quantity is logged. That log is the only way a future agent,
// which cannot see the viewer, can tell what this run produced, and it is what
// gets diffed against the Python reference in validate/.
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
            // the Python reference. Under a locale with a comma decimal
            // separator that diff turns into noise, and contour_csharp.csv
            // becomes a file where the decimal point and the field delimiter
            // are both a comma.
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

            // --exit-when-done overrides the spec, so a scripted run does not
            // need its own spec file just to avoid hanging on the viewer.
            bool bExitFlag = args.Any(a => a == "--exit-when-done");
            string[] astrPositional = args.Where(a => !a.StartsWith("--")).ToArray();
            if (astrPositional.Length > 0)
                _specPath = astrPositional[0];

            if (!File.Exists(_specPath))
            {
                Console.Error.WriteLine($"spec not found: {_specPath}");
                Environment.Exit(1);
            }

            try
            {
                EngineSpec oSpec = EngineSpec.Load(_specPath);
                Library.Go(oSpec.Geometry.VoxelSizeMm, Task,
                           bEndAppWithTask: bExitFlag || oSpec.Output.ExitWhenDone);
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
                oSpec.Nozzle.ExpansionRatio, oSpec.Nozzle.ExitRadiusMm, oSpec.Gas.Gamma,
                oSpec.Nozzle.ContourPoints, oSpec.Nozzle.TruncateFraction);

            EngineAssembly oAsm = EngineAssembly.Build(oContour, oSpec);
            ChamberConditions oChamber = ChamberConditions.FromSpec(
                oSpec, oContour.ThroatAreaMm2, oSpec.Nozzle.ExpansionRatio, oContour.ExitMach);

            LogGeometry(oAsm, oContour);
            LogPerformance(oChamber);
            CheckInvariants(oAsm, oContour, oSpec);

            InjectorDesign? oInjector = BuildInjector(oAsm, oChamber, oSpec);
            var oCircuits = BuildCooling(oAsm, oContour, oChamber, oSpec);
            CheckStructureAndStartup(oAsm, oChamber, oSpec, oCircuits);
            CheckStability(oAsm, oChamber, oSpec, oInjector);
            LogAltitudeCompensation(oContour, oChamber);

            // ---- geometry ----
            var oChannelsCowl = new List<ChannelCut>();
            var oChannelsBody = new List<ChannelCut>();
            var oPortsCowl = new List<PortCut>();
            var oPortsBody = new List<PortCut>();
            if (oCircuits.TryGetValue("cowl", out var cCowl) && cCowl is not null)
            {
                var cut = CooledFeatures.Cowl(oAsm, cCowl.Channel, oSpec.Cooling.BackWallMm);
                oChannelsCowl.Add(cut);
                oPortsCowl.Add(CooledFeatures.FeedPorts(oAsm, cut));
            }
            if (oCircuits.TryGetValue("centrebody", out var cBody) && cBody is not null)
            {
                var cut = CooledFeatures.Centrebody(oAsm, cBody.Channel, oSpec.Cooling.BackWallMm);
                oChannelsBody.Add(cut);
                oPortsBody.Add(CooledFeatures.FeedPorts(oAsm, cut));
            }

            var oHoles = oInjector is null
                ? new List<HoleCut>()
                : CooledFeatures.InjectorOrifices(oAsm, oInjector);

            WarnIfUnderResolved(oSpec, oChannelsCowl, oChannelsBody);

            // Table the profile distance well below the voxel size, so the
            // interpolation error stays far under the sampling the kernel is
            // about to do. Quarter of a voxel is ample and costs a few MB.
            double fStep = oSpec.Geometry.VoxelSizeMm / 4.0;

            Library.Log("building voxels (this is the slow part)");
            Voxels voxCentrebody = SpikeGeometry.voxBuildCooledPart(
                new CooledPart(oAsm.CentrebodyProfile(), oChannelsBody, null, oPortsBody,
                               1.0f, fStep));
            Library.Log("  centrebody done");
            Voxels voxCowl = SpikeGeometry.voxBuildCooledPart(
                new CooledPart(oAsm.CowlProfile(), oChannelsCowl, null, oPortsCowl,
                               1.0f, fStep));
            Library.Log("  cowl done");
            Voxels voxHead = SpikeGeometry.voxBuildCooledPart(
                new CooledPart(oAsm.HeadProfile(), null, oHoles, null, 1.0f, fStep));
            Library.Log("  head done");

            Sh.PreviewVoxels(voxCentrebody, Cp.clrRock, 0.9f);
            Sh.PreviewVoxels(voxCowl, Cp.clrBlue, 0.5f);
            Sh.PreviewVoxels(voxHead, Cp.clrRock, 0.7f);

            LogBox("centrebody", voxCentrebody);
            LogBox("cowl", voxCowl);
            LogBox("head", voxHead);

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

        // ------------------------------------------------------------------

        private static void LogGeometry(EngineAssembly a, PlugContour c)
        {
            Library.Log($"spec               {a.Spec.Name}");
            Library.Log($"exit Mach          {c.ExitMach:F6}");
            Library.Log($"throat angle       {c.ThroatAngleRad * 180.0 / Math.PI:F6} deg");
            Library.Log($"throat area        {c.ThroatAreaMm2:F6} mm2");
            Library.Log($"throat area (geom) {c.ThroatAreaFromGeometryMm2:F6} mm2");
            Library.Log($"lip station        {c.LipXMm:F6} mm");
            Library.Log($"throat slant       {c.ThroatSlantLengthMm:F6} mm");
            Library.Log($"spike length       {c.LengthMm:F6} mm");
            Library.Log($"tip radius         {c.TipRadiusMm:F6} mm");
            Library.Log($"shoulder radius    {a.ShoulderRadiusMm:F6} mm");
            Library.Log($"chamber outer R    {a.ChamberOuterRadiusMm:F6} mm");
            Library.Log($"contraction ratio  {a.ChamberAreaMm2 / c.ThroatAreaMm2:F6}");
            Library.Log($"bore radius        {a.BoreRadiusMm:F6} mm");
            Library.Log($"flange radius      {a.FlangeRadiusMm:F6} mm");
            Library.Log($"overall length     {a.OverallLengthMm:F6} mm");
            Library.Log($"overall diameter   {2.0 * a.OverallRadiusMm:F6} mm");
        }

        private static void LogPerformance(ChamberConditions ch)
        {
            Library.Log($"propellant         {ch.Propellant.Oxidiser}/{ch.Propellant.Fuel} at MR {ch.MixtureRatio:F3}");
            Library.Log($"chamber pressure   {ch.ChamberPressure / 1e5:F3} bar");
            Library.Log($"chamber temp       {ch.TChamber:F3} K");
            Library.Log($"c* (delivered)     {ch.CStar:F4} m/s");
            Library.Log($"mass flow          {ch.MassFlow:F6} kg/s (fuel {ch.MassFlowFuel:F6}, ox {ch.MassFlowOx:F6})");
            Library.Log($"thrust             {ch.ThrustSeaLevel:F3} N sl, {ch.ThrustVacuum:F3} N vac");
            Library.Log($"Isp                {ch.IspSeaLevel:F4} s sl, {ch.IspVacuum:F4} s vac");
            Library.Log($"exit pressure      {ch.ExitPressure / 1e5:F6} bar");
            Library.Log($"coolant capacity   {ch.CoolantCapacity / 1e3:F3} kW");
        }

        private static InjectorDesign? BuildInjector(
            EngineAssembly a, ChamberConditions ch, EngineSpec spec)
        {
            var (inj, why) = Injector.FitToFace(
                ch, a.ShoulderRadiusMm, a.ChamberOuterRadiusMm,
                spec.Injector.Stiffness, spec.Injector.DischargeCoefficient,
                spec.Injector.MinLandMm);

            Library.Log($"injector           {why}");
            if (inj is null ||
                !Injector.FitsOnFace(inj, a.ShoulderRadiusMm, a.ChamberOuterRadiusMm,
                                     spec.Injector.MinLandMm).ok)
            {
                Library.Log("WARNING no injector pattern fits the face");
                return null;
            }

            Library.Log($"  fuel orifice     {inj.DFuelMm:F4} mm x {inj.NElements} at r {inj.FuelRingRadius * 1e3:F3} mm");
            Library.Log($"  ox orifice       {inj.DOxMm:F4} mm x {inj.NElements} at r {inj.OxRingRadius * 1e3:F3} mm");
            Library.Log($"  injection dP     {inj.DpFuel / 1e5:F4} bar, chug margin {inj.ChugMargin:F3}");
            Library.Log($"  momentum ratio   {inj.MomentumRatio:F6}, resultant {inj.ResultantAngle * 180.0 / Math.PI:E3} deg");

            AssertClose(inj.ResultantAngle, 0.0, 1e-9, "doublet resultant lands on the axis");
            if (inj.ChugMargin < 1.0)
                Library.Log($"WARNING injection stiffness {inj.Stiffness:F3} is below the 0.15 chug floor");
            return inj;
        }

        /// <summary>
        /// Solve both cooling circuits.
        ///
        /// The fuel is split by measured heat load, not by a number from the
        /// spec. The centrebody carries the chamber wall and the whole plug
        /// surface, so it runs about 60/40 against the cowl, and an even split
        /// starves it while the cowl sits comfortable.
        /// </summary>
        private static Dictionary<string, CoolingCandidate?> BuildCooling(
            EngineAssembly a, PlugContour c, ChamberConditions ch, EngineSpec spec)
        {
            double dt = 2.0 * c.ThroatSlantLengthMm;   // hydraulic diameter of the slot

            FilmSpec? film = spec.Film.FuelFraction > 0.0
                ? new FilmSpec
                {
                    FuelFraction = spec.Film.FuelFraction,
                    InjectXMm = a.HeadXMm,
                    SlotHeightMm = spec.Film.SlotHeightMm,
                }
                : null;
            CoatingSpec? coating = spec.Coating.ThicknessMm > 0.0
                ? new CoatingSpec
                {
                    ThicknessMm = spec.Coating.ThicknessMm,
                    Conductivity = spec.Coating.Conductivity,
                    MaxSurfaceTemp = spec.Coating.MaxSurfaceTempK,
                }
                : null;

            var reference = new ChannelSpec
            {
                NChannels = 200, WidthMm = 0.4, HeightMm = 0.6, LandMm = 0.4, HotWallMm = 0.3,
            };
            string[] walls = { "cowl", "centrebody" };
            var load = new Dictionary<string, double>();

            foreach (string wall in walls)
            {
                var (wx, wr, war, wm) = a.WallFlowState(wall, ch.Propellant.Gamma);
                var sol = Cooling.Solve(ch, wx, wr, war, wm, reference,
                                        Cooling.Materials[spec.Cooling.Materials[0]],
                                        dt, dt, 1.0, null, film, coating);
                load[wall] = sol.TotalHeat;
            }

            double total = load.Values.Sum();
            Library.Log($"heat load          {total / 1e3:F3} kW against {ch.CoolantCapacity / 1e3:F3} kW of coolant");
            if (total > ch.CoolantCapacity)
                Library.Log("WARNING heat load exceeds what the fuel can absorb. No channel " +
                            "geometry fixes that: the engine needs more mass flow, a higher " +
                            "chamber pressure, film cooling, or to be larger");

            var result = new Dictionary<string, CoolingCandidate?>();
            foreach (string wall in walls)
            {
                double share = load[wall] / total;
                double rRef = wall == "cowl" ? a.ChamberOuterRadiusMm : a.ShoulderRadiusMm;
                var (wx, wr, war, wm) = a.WallFlowState(wall, ch.Propellant.Gamma);
                var (best, all) = Cooling.Search(
                    ch, wx, wr, war, wm, dt, dt, rRef, spec.Cooling.Materials,
                    share, spec.Cooling.MaxPressureDropFraction, 80,
                    spec.Cooling.MinChannelWidthMm, spec.Cooling.MinHotWallMm,
                    spec.Geometry.WallThicknessMm - spec.Cooling.BackWallMm,
                    film, coating);

                result[wall] = best;
                if (best is null)
                {
                    var closest = all.Aggregate((p, q) => q.Score > p.Score ? q : p);
                    Library.Log($"WARNING {wall}: no channel geometry survives on " +
                                $"{share * 100:F0}% of fuel. Closest was " +
                                $"{closest.Material.Name} -- {closest.Reason}");
                    continue;
                }
                var s = best.Solution!;
                Library.Log($"{wall,-18} {share * 100:F0}% of fuel, {best.Material.Name}, " +
                            $"{best.Channel.NChannels} x {best.Channel.WidthMm:F2}x{best.Channel.HeightMm:F2} mm, " +
                            $"{best.Channel.HotWallMm:F1} mm hot wall");
                Library.Log($"  peak flux        {s.PeakHeatFlux / 1e6:F3} MW/m2");
                Library.Log($"  wall temp        {s.PeakWallTemperature:F3} K (margin {s.WallTemperatureMargin:F3})");
                Library.Log($"  coolant out      {s.CoolantOutletTemp:F3} K (margin {s.CoolantTemperatureMargin:F3})");
                Library.Log($"  heat / drop      {s.TotalHeat / 1e3:F3} kW / {s.PressureDrop / 1e5:F4} bar");
                Library.Log($"  Reynolds         {s.MinReynolds:F0} to {s.MaxReynolds:F0}");
            }
            return result;
        }

        private static void WarnIfUnderResolved(
            EngineSpec spec, List<ChannelCut> cowl, List<ChannelCut> body)
        {
            double narrowest = double.MaxValue;
            foreach (var c in cowl.Concat(body)) narrowest = Math.Min(narrowest, c.WidthMm);
            if (narrowest == double.MaxValue) return;

            double across = narrowest / spec.Geometry.VoxelSizeMm;
            Library.Log($"narrowest channel  {narrowest:F3} mm = {across:F1} voxels across");
            if (across < 3.0)
                Library.Log($"WARNING at {spec.Geometry.VoxelSizeMm:F3} mm voxels the channels are " +
                            $"under three voxels wide and will render broken. Drop " +
                            $"geometry.voxel_size_mm to about {narrowest / 3.0:F3} mm, " +
                            $"remembering memory goes as the cube");
        }

        private static void CheckInvariants(EngineAssembly a, PlugContour c, EngineSpec spec)
        {
            AssertClose(c.ThroatAreaFromGeometryMm2, c.ThroatAreaMm2, 1e-9,
                        "inclined throat area reproduces pi*r_e^2/eps");
            AssertClose(c.ThroatInclinationRad + c.ThroatAngleRad, Math.PI / 2.0, 1e-9,
                        "sonic line sits at pi/2 - nu_e from the axis");
            AssertClose(a.OuterWallX[^1], c.LipXMm, 1e-6, "contraction lands on the lip in x");
            AssertClose(a.OuterWallR[^1], c.ExitRadiusMm, 1e-6, "contraction lands on the lip in r");

            if (spec.Nozzle.TruncateFraction >= 1.0)
                Assert(c.TipRadiusMm < 0.05 * c.ExitRadiusMm,
                       "ideal full-length spike closes on the axis");

            var (fMinX, fMinArea) = a.MinimumFlowArea();
            Library.Log($"min flow area      {fMinArea:F6} mm2 at x={fMinX:F6} mm");
            AssertClose(fMinArea / c.ThroatAreaMm2, 1.0, 2e-3,
                        "minimum duct area equals the throat area");
            AssertClose(fMinX, c.LipXMm, 0.05,
                        "the choke point is at the throat, not upstream of it");
        }

        /// <summary>
        /// Thermal strain and startup, per circuit.
        ///
        /// Both are life-or-death checks that the steady thermal solution says
        /// nothing about: the wall yields every firing and its life is set by
        /// how much plastic strain it takes, and the whole design can be lost in
        /// the first half second if the coolant arrives after the gas.
        /// </summary>
        private static void CheckStructureAndStartup(
            EngineAssembly a, ChamberConditions ch, EngineSpec spec,
            Dictionary<string, CoolingCandidate?> circuits)
        {
            foreach (var (name, cand) in circuits)
            {
                if (cand?.Solution is null) continue;
                CoolingSolution sol = cand.Solution;

                var st = Structural.Analyse(sol, ch.ChamberPressure,
                                            1.5 * ch.ChamberPressure, sol.WallRadiusMm,
                                            spec.Geometry.WallThicknessMm,
                                            spec.Structure.RequiredCycles);
                Library.Log($"{name,-18} thermal {st.ThermalStress.Max() / 1e6:F2} MPa, " +
                            $"hoop {st.HoopStress.Max() / 1e6:F2}, " +
                            $"combined {st.CombinedStress.Max() / 1e6:F2} MPa");
                Library.Log($"  yields           {(st.StaysElastic ? "no" : "yes, as expected")}");
                Library.Log($"  cycle life       {st.CyclesToFailure:F0} against {st.RequiredCycles:F0} required");
                if (!st.Survives)
                    Library.Log($"WARNING {name}: structural life is short of what is required");

                var run = Transient.SolveStartup(sol, null, spec.Startup.RampTimeS,
                                                 spec.Startup.CoolantLeadS);
                Library.Log($"  startup peak     {run.PeakWallTemp:F2} K vs steady " +
                            $"{run.SteadyWallTemp:F2} K, overshoot {run.Overshoot:F2} K");
                Library.Log($"  wall diffusion   {run.DiffusionTime * 1e3:F3} ms");
                if (!run.Survives)
                    Library.Log($"WARNING {name}: startup peaks at {run.PeakWallTemp:F0} K, over " +
                                $"the {run.Material.MaxWallTemp:F0} K limit, with a coolant lead " +
                                $"of {spec.Startup.CoolantLeadS:+0.00;-0.00} s");
            }
        }

        /// <summary>
        /// Where the chamber wants to ring, and whether the standard screens
        /// pass. This cannot say the engine is stable; only hot fire can.
        /// </summary>
        private static void CheckStability(EngineAssembly a, ChamberConditions ch,
                                           EngineSpec spec, InjectorDesign? inj)
        {
            var s = Stability.Analyse(
                ch,
                chamberLengthM: (a.ConvergingStartXMm - a.HeadXMm) * 1e-3,
                meanRadiusM: 0.5 * (a.ShoulderRadiusMm + a.ChamberOuterRadiusMm) * 1e-3,
                annulusGapM: (a.ChamberOuterRadiusMm - a.ShoulderRadiusMm) * 1e-3,
                injectionStiffness: inj?.Stiffness ?? 0.0);

            Library.Log($"speed of sound     {s.SpeedOfSound:F3} m/s");
            Library.Log($"L*                 {s.CharacteristicLength:F6} m");
            Library.Log($"residence time     {s.ResidenceTime * 1e3:F4} ms");
            Library.Log($"1L / 1T / 2T / 1R  {s.FLongitudinal1:F0} / {s.FTangential1:F0} / " +
                        $"{s.FTangential2:F0} / {s.FRadial1:F0} Hz");
            Library.Log($"chug estimate      {s.FChugEstimate:F0} Hz");

            if (!s.LStarOk)
                Library.Log($"WARNING L* of {s.CharacteristicLength:F3} m is outside the 0.6 to " +
                            $"1.5 m band: too short and the propellant leaves before it has " +
                            $"burned, so the assumed c* efficiency is optimistic");
            if (!s.SeparationOk)
                Library.Log($"WARNING chug estimate {s.FChugEstimate:F0} Hz is not well separated " +
                            $"from the first tangential at {s.FTangential1:F0} Hz");
            if (!s.StiffnessOk)
                Library.Log("WARNING injection stiffness is below the 0.15 chug floor");
        }

        /// <summary>
        /// Thrust from the plug surface pressure against ambient, which is the
        /// altitude compensation the ideal bell coefficient cannot show.
        /// </summary>
        private static void LogAltitudeCompensation(PlugContour c, ChamberConditions ch)
        {
            double eps = Math.PI * c.ExitRadiusMm * c.ExitRadiusMm / c.ThroatAreaMm2;
            foreach (double pa in new[] { 1.01325e5, 0.5e5, 0.2e5, 0.05e5, 0.0 })
            {
                var t = PlugFlow.Thrust(c, ch, pa);
                double bell = Combustion.ThrustCoefficient(ch.Propellant.Gamma, eps,
                                                           c.ExitMach, ch.ChamberPressure, pa);
                Library.Log($"pa {pa / 1e5,7:F3} bar   CF plug {t.ThrustCoefficient:F4}   " +
                            $"CF bell {bell:F4}   gain {t.ThrustCoefficient / bell:F3}   " +
                            $"attached {t.AttachedFraction * 100:F1}%   Isp {t.Isp:F2} s");
            }
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
