namespace SharpAI.Domain.Interfaces;

public interface IModel
{
    double[] Predict(double[][] inputs);
}
