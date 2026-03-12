namespace SharpAI.Domain.Interfaces;

public interface IReinforcementLearner
{
    int SelectAction(double[] state);
    void Update(double[] state, int action, double reward, double[] nextState, bool done);
    void Reset();
}
