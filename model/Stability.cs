//
// Combustion stability screening.
//
// Port of validate/stability_ref.py. That file is authoritative.
//
// Stability cannot be predicted from geometry. What can be done, and what every
// preliminary design does, is compute the frequencies the chamber supports and
// the classic screening parameters, then keep the injector away from them and
// test. Chug is guarded with injection stiffness; the first tangential is the
// mode that destroys hardware, and in an annular chamber its wavelength is set
// by the mean circumference rather than by the gap, which puts it lower than a
// cylindrical chamber of the same volume would.
//
// No Rayleigh criterion, no combustion response function, no eigenvalue
// problem, no nonlinear triggering. This says where the chamber wants to ring.
// It cannot say the engine is stable, and nothing short of hot fire can.
//

namespace AerospikeCE
{
    public sealed class StabilityResult
    {
        public double SpeedOfSound;
        public double ChamberVolume;
        public double CharacteristicLength;   // L* = V_c / A_t
        public double ResidenceTime;
        public double FLongitudinal1;
        public double FTangential1;
        public double FTangential2;
        public double FRadial1;
        public double FChugEstimate;
        public double InjectionStiffness;
        public bool LStarOk;
        public bool StiffnessOk;
        public bool SeparationOk;

        public bool ScreensPass => LStarOk && StiffnessOk && SeparationOk;
    }

    public static class Stability
    {
        public static StabilityResult Analyse(
            ChamberConditions chamber, double chamberLengthM, double meanRadiusM,
            double annulusGapM, double injectionStiffness,
            double lStarMin = 0.6, double lStarMax = 1.5)
        {
            Propellant p = chamber.Propellant;
            double rGas = Combustion.RUniversal / p.MolarMass;
            double c = Math.Sqrt(p.Gamma * rGas * chamber.TChamber);

            double volume = 2.0 * Math.PI * meanRadiusM * annulusGapM * chamberLengthM;
            double lStar = volume / chamber.ThroatArea;
            double rhoC = chamber.ChamberPressure / (rGas * chamber.TChamber);
            double residence = rhoC * volume / chamber.MassFlow;

            double f1t = c / (2.0 * Math.PI * meanRadiusM);
            double fChug = 1.0 / (2.0 * Math.PI * residence);

            return new StabilityResult
            {
                SpeedOfSound = c,
                ChamberVolume = volume,
                CharacteristicLength = lStar,
                ResidenceTime = residence,
                FLongitudinal1 = c / (2.0 * chamberLengthM),
                FTangential1 = f1t,
                FTangential2 = 2.0 * f1t,
                FRadial1 = c / (2.0 * annulusGapM),
                FChugEstimate = fChug,
                InjectionStiffness = injectionStiffness,
                LStarOk = lStar >= lStarMin && lStar <= lStarMax,
                StiffnessOk = injectionStiffness >= 0.15,
                SeparationOk = fChug < 0.2 * f1t,
            };
        }
    }
}
