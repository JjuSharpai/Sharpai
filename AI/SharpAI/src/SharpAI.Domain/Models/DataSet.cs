namespace SharpAI.Domain.Models;

public class DataSet
{
    public double[][] Features { get; }
    public double[]? Labels { get; }

    public DataSet(double[][] features, double[]? labels = null)
    {
        Features = features;
        Labels = labels;
    }
}
