namespace SharpAI.Domain.Interfaces.Training;

public interface ISupervised
{
    void Train(double[][] features, double[] labels);
    double[] Predict(double[][] inputs);
    double Evaluate(double[][] features, double[] labels);
}
