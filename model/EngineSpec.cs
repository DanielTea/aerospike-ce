//
// The descriptive input. Nothing in this project is drawn by hand; everything is
// derived from this file.
//

using System.Text.Json;
using System.Text.Json.Serialization;

namespace AerospikeCE
{
    public sealed class GasSpec
    {
        [JsonPropertyName("gamma")] public double Gamma { get; set; } = 1.20;
    }

    public sealed class NozzleSpec
    {
        [JsonPropertyName("expansion_ratio")]   public double ExpansionRatio { get; set; } = 8.0;
        [JsonPropertyName("exit_radius_mm")]    public double ExitRadiusMm { get; set; } = 30.0;
        [JsonPropertyName("contour_points")]    public int ContourPoints { get; set; } = 300;
        [JsonPropertyName("truncate_fraction")] public double TruncateFraction { get; set; } = 1.0;
    }

    /// <summary>
    /// Annular chamber geometry. Ratios and lengths only -- no combustion, no
    /// propellant, no injector. See CLAUDE.md on what stays out of this model.
    /// </summary>
    public sealed class ChamberSpec
    {
        [JsonPropertyName("contraction_ratio")]    public double ContractionRatio { get; set; } = 3.0;
        [JsonPropertyName("chamber_length_mm")]    public double ChamberLengthMm { get; set; } = 18.0;
        [JsonPropertyName("converging_length_mm")] public double ConvergingLengthMm { get; set; } = 14.0;
    }

    public sealed class GeometrySpec
    {
        [JsonPropertyName("voxel_size_mm")]          public float VoxelSizeMm { get; set; } = 0.2f;
        [JsonPropertyName("wall_thickness_mm")]      public double WallThicknessMm { get; set; } = 2.0;
        [JsonPropertyName("lip_thickness_mm")]       public double LipThicknessMm { get; set; } = 0.8;
        [JsonPropertyName("base_cap_thickness_mm")]  public double BaseCapThicknessMm { get; set; } = 3.0;
        [JsonPropertyName("head_thickness_mm")]      public double HeadThicknessMm { get; set; } = 4.0;
        [JsonPropertyName("flange_width_mm")]        public double FlangeWidthMm { get; set; } = 8.0;
    }

    public sealed class OutputSpec
    {
        [JsonPropertyName("stl_dir")]     public string StlDir { get; set; } = "out";
        [JsonPropertyName("export_spike")] public bool ExportSpike { get; set; } = true;
        [JsonPropertyName("export_cowl")]  public bool ExportCowl { get; set; } = true;
        [JsonPropertyName("export_head")]  public bool ExportHead { get; set; } = true;
    }

    public sealed class EngineSpec
    {
        [JsonPropertyName("name")]     public string Name { get; set; } = "unnamed";
        [JsonPropertyName("gas")]      public GasSpec Gas { get; set; } = new();
        [JsonPropertyName("nozzle")]   public NozzleSpec Nozzle { get; set; } = new();
        [JsonPropertyName("chamber")]  public ChamberSpec Chamber { get; set; } = new();
        [JsonPropertyName("geometry")] public GeometrySpec Geometry { get; set; } = new();
        [JsonPropertyName("output")]   public OutputSpec Output { get; set; } = new();

        public static EngineSpec Load(string path)
        {
            string json = File.ReadAllText(path);
            var opts = new JsonSerializerOptions { ReadCommentHandling = JsonCommentHandling.Skip };
            return JsonSerializer.Deserialize<EngineSpec>(json, opts)
                   ?? throw new Exception($"could not parse spec: {path}");
        }
    }
}
