namespace SharpAI.Domain.Interfaces;

public interface IEnvironment
{
    int StateSize { get; }
    int ActionCount { get; }
    double[] Reset();
    (double[] NextState, double Reward, bool Done) Step(int action);
}
