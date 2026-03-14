namespace SharpAI.Domain.Interfaces.Training;

public interface IReinforcement
{
    int SelectAction(double[] state);
    void Update(double[] state, int action, double reward, double[] nextState, bool done);
    void Reset();
}
