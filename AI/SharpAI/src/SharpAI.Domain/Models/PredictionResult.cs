namespace SharpAI.Domain.Models;

public class PredictionResult
{
    public required double[] Predictions { get; init; }
    public IDictionary<string, double>? Metrics { get; init; }
}
