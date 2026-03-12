namespace SharpAI.Domain.Interfaces;

public interface ISupervisedLearner
{
    IModel Train(double[][] features, double[] labels);
    double Evaluate(IModel model, double[][] features, double[] labels);
}
