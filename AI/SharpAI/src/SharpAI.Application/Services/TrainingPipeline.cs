using SharpAI.Domain.Interfaces;
using SharpAI.Domain.Interfaces.Training;
using SharpAI.Domain.Models;

namespace SharpAI.Application.Services;

public class TrainingPipeline
{
    public PredictionResult RunSupervised(ISupervised learner, DataSet trainingSet, DataSet testSet)
    {
        learner.Train(trainingSet.Features, trainingSet.Labels!);
        var predictions = learner.Predict(testSet.Features);
        var accuracy = learner.Evaluate(testSet.Features, testSet.Labels!);

        return new PredictionResult
        {
            Predictions = predictions,
            Metrics = new Dictionary<string, double> { ["accuracy"] = accuracy }
        };
    }

    public PredictionResult RunUnsupervised(IUnsupervised learner, DataSet dataSet)
    {
        learner.Train(dataSet.Features);
        var assignments = learner.Predict(dataSet.Features);

        return new PredictionResult { Predictions = assignments };
    }

    public void RunReinforcement(IReinforcement agent, IEnvironment env, int episodes)
    {
        for (int ep = 0; ep < episodes; ep++)
        {
            var state = env.Reset();
            agent.Reset();
            bool done = false;

            while (!done)
            {
                int action = agent.SelectAction(state);
                var (nextState, reward, isDone) = env.Step(action);
                agent.Update(state, action, reward, nextState, isDone);
                state = nextState;
                done = isDone;
            }
        }
    }
}
