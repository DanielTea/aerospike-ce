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

    public sealed class GeometrySpec
    {
        [JsonPropertyName("voxel_size_mm")]           public float VoxelSizeMm { get; set; } = 0.2f;
        [JsonPropertyName("cowl_thickness_mm")]       public float CowlThicknessMm { get; set; } = 3.0f;
        [JsonPropertyName("cowl_length_mm")]          public float CowlLengthMm { get; set; } = 12.0f;
        [JsonPropertyName("base_plate_thickness_mm")] public float BasePlateThicknessMm { get; set; } = 4.0f;
    }

    public sealed class OutputSpec
    {
        [JsonPropertyName("stl_dir")]      public string StlDir { get; set; } = "out";
        [JsonPropertyName("export_spike")] public bool ExportSpike { get; set; } = true;
        [JsonPropertyName("export_cowl")]  public bool ExportCowl { get; set; } = true;
    }

    public sealed class EngineSpec
    {
        [JsonPropertyName("name")]     public string Name { get; set; } = "unnamed";
        [JsonPropertyName("gas")]      public GasSpec Gas { get; set; } = new();
        [JsonPropertyName("nozzle")]   public NozzleSpec Nozzle { get; set; } = new();
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
