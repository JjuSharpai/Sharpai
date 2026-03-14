namespace SharpAI.Domain.Interfaces.Training;

public interface IUnsupervised
{
    void Train(double[][] features);
    double[] Predict(double[][] inputs);
}
