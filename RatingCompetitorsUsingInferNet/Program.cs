using System;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;  // 49 dependencies
// determine the "Strength" of each of 6 teams based on 9 games total
// Graphviz "dot" program optional
// https://graphviz.gitlab.io/_pages/Download/Download_windows.html
// VS2017 (Framework 4.7) Infer.NET 0.3.1810.501

namespace RatingCompetitorsUsingInferNet
{
    class InferStrengthsProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin Infer.NET demo \n");

            // ===== set up teams and win-loss data =========================

            string[] teamNames = new string[] { "Angels", "Bruins", "Comets",
        "Demons", "Eagles", "Flyers" };
            int N = teamNames.Length;  // number teams

            int[] winTeamIDs = new int[] { 0, 2, 1, 0, 1, 3, 0, 2, 4 };
            int[] loseTeamIDs = new int[] { 1, 3, 2, 4, 3, 5, 5, 4, 5 };

            //double[] initStrengths = new double[] { 2500, 1900, 2000, 1700,
            //  2400, 2000 };
            //Console.WriteLine("Initial team strengths: \n");
            //for (int i = 0; i < N; ++i)
            //  Console.WriteLine(teamNames[i] + ": " +
            //    initStrengths[i].ToString("F1"));
            //Console.WriteLine("");

            Console.WriteLine("Data: \n");
            for (int i = 0; i < winTeamIDs.Length; ++i)
            {
                Console.WriteLine("game: " + i + "   winning team: " +
                  teamNames[winTeamIDs[i]] + "   losing team: " +
                  teamNames[loseTeamIDs[i]]);
            }

            // ===== define a probabilistic model ===========================

            Range teamIDsRange = new Range(N).Named("teamsIDRange");
            Range gameIDsRange =
              new Range(winTeamIDs.Length).Named("gameIDsRange");

            double mean = 5.0;
            double sd = 5.0;
            double vrnc = sd * sd;
            // double precision = 1.0 / (sd * sd);

            Console.WriteLine("\nDefining Gaussian model with mean = " +
              mean.ToString("F1") + " and sd = " + sd.ToString("F1"));
            VariableArray<double> strengths =
              Variable.Array<double>(teamIDsRange).Named("strengths");
            //teamStrengths[teamIDsRange] =
            //  Variable.GaussianFromMeanAndPrecision(mean,
            //    precision).ForEach(teamIDsRange);
            strengths[teamIDsRange] =
              Variable.GaussianFromMeanAndVariance(mean,
              vrnc).ForEach(teamIDsRange);

            //Gaussian[] inits = new Gaussian[teamIDsRange.SizeAsInt];
            //for (int i = 0; i < inits.Length; ++i)
            //  inits[i] = Gaussian.FromMeanAndVariance(initStrengths[i], 100.0);
            //VariableArray<Gaussian> initVar =
            //  Variable.Observed(inits, teamIDsRange).Named("initVar");
            //teamStrengths[teamIDsRange].InitialiseTo(initVar[teamIDsRange]);

            VariableArray<int> winners =
              Variable.Array<int>(gameIDsRange).Named("winners");
            VariableArray<int> losers =
              Variable.Array<int>(gameIDsRange).Named("losers");

            winners.ObservedValue = winTeamIDs;
            losers.ObservedValue = loseTeamIDs;

            using (Variable.ForEach(gameIDsRange))
            {
                var ws = strengths[winners[gameIDsRange]];
                var ls = strengths[losers[gameIDsRange]];
                Variable<double> winPerf =
                  Variable.GaussianFromMeanAndVariance(ws, 400.0).Named("winPerf");
                Variable<double> losePerf =
                  Variable.GaussianFromMeanAndVariance(ls, 400.0).Named("losePerf");


                Variable.ConstrainTrue(winPerf > losePerf);

                //Variable.ConstrainPositive(winnerPerf - loserPerf);
            }

            // ===== infer team strengths using win-loss data ===============

            Console.WriteLine("\nInferring strengths from win-loss data \n");
            var iengine = new InferenceEngine();
            iengine.Algorithm = new ExpectationPropagation();
            iengine.NumberOfIterations = 40;
            // iengine.ShowFactorGraph = true;  // needs Graphviz

            Gaussian[] inferredStrengths =
              iengine.Infer<Gaussian[]>(strengths);
            Console.WriteLine("\nInference complete. Inferred strengths: \n");

            // ===== show results ===========================================

            for (int i = 0; i < N; ++i)
            {
                double strength = inferredStrengths[i].GetMean();
                Console.WriteLine(teamNames[i] + ": " + strength.ToString("F1"));
            }

            Console.WriteLine("\nEnd demo ");
            Console.ReadLine();
        } // Main
    }
}
