using SharpAI.Domain.Interfaces;
using SharpAI.Domain.Models;

namespace SharpAI.Application.Services;

public class TrainingPipeline
{
    public PredictionResult RunSupervised(ISupervisedLearner learner, DataSet trainingSet, DataSet testSet)
    {
        var model = learner.Train(trainingSet.Features, trainingSet.Labels!);
        var predictions = model.Predict(testSet.Features);
        var accuracy = learner.Evaluate(model, testSet.Features, testSet.Labels!);

        return new PredictionResult
        {
            Predictions = predictions,
            Metrics = new Dictionary<string, double> { ["accuracy"] = accuracy }
        };
    }

    public PredictionResult RunUnsupervised(IUnsupervisedLearner learner, DataSet dataSet)
    {
        var model = learner.Train(dataSet.Features);
        var assignments = model.Predict(dataSet.Features);

        return new PredictionResult { Predictions = assignments };
    }

    public void RunReinforcement(IReinforcementLearner agent, IEnvironment env, int episodes)
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
