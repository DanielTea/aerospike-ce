//
// The build plan: what Python decided, read back for voxelisation.
//
// This file is the whole of the C# side's knowledge of the engine. It knows
// what shapes to build and where; it knows nothing about why. There is no
// contour construction here, no gas dynamics, no cooling solve -- those live in
// validate/ and have exactly one implementation.
//
// The arrangement replaced a mirror. Two thirds of this project's C# used to be
// a hand-kept copy of the Python maths, with divergence named in CLAUDE.md as
// the failure mode the structure existed to prevent. It never caught an error
// in the Python; every bug the pairing found was a bug in the copy. A plan file
// removes the duplication rather than policing it.
//
// Read strictly. A plan whose version is not understood is refused rather than
// guessed at, because a silently misread plan builds a plausible engine that is
// not the one that was designed.
//

using System.Text.Json;
using System.Text.Json.Serialization;

namespace AerospikeCE
{
    public sealed class PlanPoints
    {
        [JsonPropertyName("x")] public double[] X { get; set; } = Array.Empty<double>();
        [JsonPropertyName("r")] public double[] R { get; set; } = Array.Empty<double>();
    }

    public sealed class PlanChannel
    {
        [JsonPropertyName("wall")]        public PlanPoints Wall { get; set; } = new();
        [JsonPropertyName("count")]       public int Count { get; set; }
        [JsonPropertyName("width_mm")]    public double WidthMm { get; set; }
        [JsonPropertyName("height_mm")]   public double HeightMm { get; set; }
        [JsonPropertyName("hot_wall_mm")] public double HotWallMm { get; set; }
        [JsonPropertyName("land_mm")]     public double LandMm { get; set; }
        [JsonPropertyName("x_start_mm")]  public double XStartMm { get; set; }
        [JsonPropertyName("x_end_mm")]    public double XEndMm { get; set; }
        [JsonPropertyName("outward")]     public bool Outward { get; set; }
    }

    public sealed class PlanPort
    {
        [JsonPropertyName("x_mm")]        public double XMm { get; set; }
        [JsonPropertyName("diameter_mm")] public double DiameterMm { get; set; }
        [JsonPropertyName("count")]       public int Count { get; set; }
        [JsonPropertyName("r_lo_mm")]     public double RLoMm { get; set; }
        [JsonPropertyName("r_hi_mm")]     public double RHiMm { get; set; }
        [JsonPropertyName("phase_rad")]   public double PhaseRad { get; set; }
    }

    public sealed class PlanHole
    {
        [JsonPropertyName("radius_mm")]   public double RadiusMm { get; set; }
        [JsonPropertyName("diameter_mm")] public double DiameterMm { get; set; }
        [JsonPropertyName("count")]       public int Count { get; set; }
        [JsonPropertyName("x_start_mm")]  public double XStartMm { get; set; }
        [JsonPropertyName("x_end_mm")]    public double XEndMm { get; set; }
        [JsonPropertyName("phase_rad")]   public double PhaseRad { get; set; }
    }

    public sealed class PlanBoss
    {
        [JsonPropertyName("x_mm")]       public double XMm { get; set; }
        [JsonPropertyName("r_inner_mm")] public double RInnerMm { get; set; }
        [JsonPropertyName("r_outer_mm")] public double ROuterMm { get; set; }
        [JsonPropertyName("half_x_mm")]  public double HalfXMm { get; set; }
    }

    public sealed class PlanPlenum
    {
        [JsonPropertyName("x_mm")]       public double XMm { get; set; }
        [JsonPropertyName("r_inner_mm")] public double RInnerMm { get; set; }
        [JsonPropertyName("half_x_mm")]  public double HalfXMm { get; set; }
        [JsonPropertyName("half_r_mm")]  public double HalfRMm { get; set; }
    }

    public sealed class PlanLug
    {
        [JsonPropertyName("count")]          public int Count { get; set; }
        [JsonPropertyName("x_mm")]           public double XMm { get; set; }
        [JsonPropertyName("half_x_mm")]      public double HalfXMm { get; set; }
        [JsonPropertyName("r_inner_mm")]     public double RInnerMm { get; set; }
        [JsonPropertyName("r_outer_mm")]     public double ROuterMm { get; set; }
        [JsonPropertyName("half_width_deg")] public double HalfWidthDeg { get; set; }
        [JsonPropertyName("phase_rad")]      public double PhaseRad { get; set; }
    }

    public sealed class PlanPart
    {
        [JsonPropertyName("profile")]  public PlanPoints Profile { get; set; } = new();
        [JsonPropertyName("channels")] public List<PlanChannel> Channels { get; set; } = new();
        [JsonPropertyName("ports")]    public List<PlanPort> Ports { get; set; } = new();
        [JsonPropertyName("holes")]    public List<PlanHole> Holes { get; set; } = new();
        [JsonPropertyName("bosses")]   public List<PlanBoss> Bosses { get; set; } = new();
        [JsonPropertyName("lugs")]     public List<PlanLug> Lugs { get; set; } = new();
        [JsonPropertyName("plenums")]  public List<PlanPlenum> Plenums { get; set; } = new();
    }

    public sealed class PlanSummary
    {
        [JsonPropertyName("thrust_sea_level_n")]   public double ThrustSeaLevelN { get; set; }
        [JsonPropertyName("thrust_vacuum_n")]      public double ThrustVacuumN { get; set; }
        [JsonPropertyName("isp_sea_level_s")]      public double IspSeaLevelS { get; set; }
        [JsonPropertyName("isp_vacuum_s")]         public double IspVacuumS { get; set; }
        [JsonPropertyName("chamber_pressure_bar")] public double ChamberPressureBar { get; set; }
        [JsonPropertyName("overall_length_mm")]    public double OverallLengthMm { get; set; }
        [JsonPropertyName("overall_diameter_mm")]  public double OverallDiameterMm { get; set; }
    }

    public sealed class BuildPlan
    {
        public const int SupportedVersion = 1;

        [JsonPropertyName("plan_version")]   public int PlanVersion { get; set; }
        [JsonPropertyName("name")]           public string Name { get; set; } = "engine";
        [JsonPropertyName("units")]          public string Units { get; set; } = "mm";
        [JsonPropertyName("voxel_size_mm")]  public double VoxelSizeMm { get; set; } = 0.2;
        [JsonPropertyName("build_direction")] public string BuildDirection { get; set; } = "+x";
        [JsonPropertyName("parts")]          public Dictionary<string, PlanPart> Parts { get; set; } = new();
        [JsonPropertyName("summary")]        public PlanSummary Summary { get; set; } = new();

        public static BuildPlan Load(string path)
        {
            var opts = new JsonSerializerOptions { ReadCommentHandling = JsonCommentHandling.Skip };
            BuildPlan plan = JsonSerializer.Deserialize<BuildPlan>(File.ReadAllText(path), opts)
                             ?? throw new Exception($"could not parse plan: {path}");

            if (plan.PlanVersion != SupportedVersion)
                throw new Exception(
                    $"plan version {plan.PlanVersion}, this build understands " +
                    $"{SupportedVersion}. Refusing rather than guessing: a misread plan " +
                    $"builds a plausible engine that is not the designed one.");
            if (plan.Units != "mm")
                throw new Exception($"plan is in {plan.Units}; this build assumes mm");
            if (plan.Parts.Count == 0)
                throw new Exception("plan contains no parts");
            foreach (var (name, part) in plan.Parts)
            {
                if (part.Profile.X.Length < 3 || part.Profile.X.Length != part.Profile.R.Length)
                    throw new Exception($"part '{name}' has a malformed profile");
            }
            return plan;
        }

        /// <summary>
        /// The smallest dimension anything in the plan has.
        ///
        /// Voxel size has to resolve it, about three samples across. Coarser
        /// and the cooling channels come out broken -- silently, because a part
        /// with its channels missing still meshes and still looks like an
        /// engine.
        /// </summary>
        public double NarrowestFeatureMm()
        {
            double smallest = double.MaxValue;
            foreach (var part in Parts.Values)
            {
                foreach (var c in part.Channels)
                    smallest = Math.Min(smallest, Math.Min(c.WidthMm,
                                        Math.Min(c.HeightMm, c.HotWallMm)));
                foreach (var h in part.Holes) smallest = Math.Min(smallest, h.DiameterMm);
                foreach (var p in part.Ports) smallest = Math.Min(smallest, p.DiameterMm);
            }
            return smallest == double.MaxValue ? 0.0 : smallest;
        }

        /// <summary>Build the implicit for one part, features and all.</summary>
        public CooledPart PartImplicit(string name, double tableStepMm)
        {
            PlanPart p = Parts[name];

            var channels = p.Channels.Select(c => new ChannelCut
            {
                WallX = c.Wall.X, WallR = c.Wall.R, NChannels = c.Count,
                WidthMm = c.WidthMm, HeightMm = c.HeightMm,
                HotWallMm = c.HotWallMm, LandMm = c.LandMm,
                XStart = c.XStartMm, XEnd = c.XEndMm, Outward = c.Outward,
            }).ToList();

            var ports = p.Ports.Select(x => new PortCut
            {
                XAt = x.XMm, DiameterMm = x.DiameterMm, Count = x.Count,
                RLo = x.RLoMm, RHi = x.RHiMm, Phase = x.PhaseRad,
            }).ToList();

            var holes = p.Holes.Select(h => new HoleCut
            {
                RadiusMm = h.RadiusMm, DiameterMm = h.DiameterMm, Count = h.Count,
                XStart = h.XStartMm, XEnd = h.XEndMm, Phase = h.PhaseRad,
            }).ToList();

            var bosses = p.Bosses.Select(b => new RingBoss
            {
                XAt = b.XMm, RInner = b.RInnerMm, ROuter = b.ROuterMm, HalfX = b.HalfXMm,
            }).ToList();

            var lugs = p.Lugs.Select(l => new LugAdd
            {
                Count = l.Count, XAt = l.XMm, HalfX = l.HalfXMm,
                RInner = l.RInnerMm, ROuter = l.ROuterMm,
                HalfWidthDeg = l.HalfWidthDeg, Phase = l.PhaseRad,
            }).ToList();

            var plenums = p.Plenums.Select(m => new PlenumCut
            {
                XAt = m.XMm, RInner = m.RInnerMm, HalfX = m.HalfXMm, HalfR = m.HalfRMm,
            }).ToList();

            return new CooledPart((p.Profile.X, p.Profile.R), channels, holes, ports,
                                  bosses, lugs, plenums, 1.0f, tableStepMm);
        }
    }
}
