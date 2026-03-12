namespace SharpAI.Domain.Interfaces;

public interface IUnsupervisedLearner
{
    IModel Train(double[][] features);
}
