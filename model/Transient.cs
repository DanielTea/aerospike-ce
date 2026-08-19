//
// Startup transient through the cooled wall.
//
// Port of validate/transient_ref.py. That file is authoritative.
//
// The steady solution says where the wall settles. It says nothing about the
// first half second, which is when regen chambers are actually lost.
//
// A thin copper hot wall diffuses heat in about ten milliseconds, far faster
// than any chamber pressure ramp, so the wall itself has no meaningful lag. The
// risk is sequencing: if the propellant valves open before the coolant is
// primed, the wall sees rising flux with nothing behind it, and copper's
// conductivity becomes a liability. This marches an explicit one-dimensional
// conduction solution with the gas and coolant sides ramping on their own
// schedules, so the cost of a coolant lead or lag reads off directly.
//
// Not modelled: axial conduction, jacket heat capacity, ignition overpressure,
// or two-phase coolant during priming.
//

namespace AerospikeCE
{
    public sealed class StartupResult
    {
        public double[] Time = Array.Empty<double>();
        public double[] WallGasTemp = Array.Empty<double>();
        public double PeakWallTemp;
        public double PeakTime;
        public double SteadyWallTemp;
        public double DiffusionTime;
        public Material Material = null!;

        public double Overshoot => PeakWallTemp - SteadyWallTemp;
        public bool Survives => PeakWallTemp < Material.MaxWallTemp;
    }

    public static class Transient
    {
        /// <summary>
        /// March the hot wall through startup at one station.
        ///
        /// coolantLead is how far ahead of the gas the coolant is established.
        /// Positive means coolant first, which is what a sane start sequence
        /// does. Negative means gas first, and is the case worth looking at
        /// before writing a valve sequence.
        /// </summary>
        public static StartupResult SolveStartup(
            CoolingSolution sol, int? station = null, double rampTime = 0.30,
            double coolantLead = 0.0, double duration = 2.0, int nodes = 41,
            double safety = 0.25)
        {
            Material m = sol.Material;
            int i = station ?? Array.IndexOf(sol.HeatFlux, sol.HeatFlux.Max());

            double tHot = sol.Channel.HotWallMm * 1e-3;
            double dx = tHot / (nodes - 1);
            double alpha = m.Conductivity / (m.Density * 385.0);
            double dt = safety * dx * dx / (2.0 * alpha);
            int steps = Math.Max(2, (int)(duration / dt));

            double qSteady = sol.HeatFlux[i];
            double tAw = sol.TAdiabatic.Length > 0 ? sol.TAdiabatic[i] : sol.TWallGas[i];
            double tCool = sol.TCoolant[i];
            double hCool = qSteady / Math.Max(sol.TWallCoolant[i] - tCool, 1e-6);
            double hGas = qSteady / Math.Max(tAw - sol.TWallGas[i], 1e-6);

            var temp = Enumerable.Repeat(tCool, nodes).ToArray();
            var times = new double[steps];
            var tg = new double[steps];

            for (int k = 0; k < steps; k++)
            {
                double t = k * dt;
                double gasFrac = Math.Clamp(t / rampTime, 0.0, 1.0);
                double coolFrac = Math.Clamp((t + coolantLead) / rampTime, 0.0, 1.0);

                double qIn = hGas * gasFrac * (tAw - temp[0]);
                double qOut = hCool * coolFrac * (temp[nodes - 1] - tCool);

                var next = (double[])temp.Clone();
                for (int j = 1; j < nodes - 1; j++)
                    next[j] = temp[j] + alpha * dt / (dx * dx)
                              * (temp[j + 1] - 2.0 * temp[j] + temp[j - 1]);
                next[0] = temp[0] + 2.0 * alpha * dt / (dx * dx)
                          * (temp[1] - temp[0] + qIn * dx / m.Conductivity);
                next[nodes - 1] = temp[nodes - 1] + 2.0 * alpha * dt / (dx * dx)
                          * (temp[nodes - 2] - temp[nodes - 1] - qOut * dx / m.Conductivity);
                temp = next;

                times[k] = t;
                tg[k] = temp[0];
            }

            double peak = tg.Max();
            return new StartupResult
            {
                Time = times, WallGasTemp = tg,
                PeakWallTemp = peak, PeakTime = times[Array.IndexOf(tg, peak)],
                SteadyWallTemp = sol.TWallGas[i],
                DiffusionTime = tHot * tHot / alpha,
                Material = m,
            };
        }
    }
}
