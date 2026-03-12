using SharpAI.Domain.Interfaces;

namespace SharpAI.Infrastructure.SupervisedLearning;

/// <summary>
/// Linear SVM 학습기 (One-vs-One 멀티클래스 전략, SGD 기반).
/// </summary>
public class SvmLearner : ISupervisedLearner
{
    private readonly double _learningRate;
    private readonly double _lambda;
    private readonly int _epochs;

    public SvmLearner(double learningRate = 0.001, double lambda = 0.01, int epochs = 1000)
    {
        _learningRate = learningRate;
        _lambda = lambda;
        _epochs = epochs;
    }

    public IModel Train(double[][] features, double[] labels)
    {
        var classes = labels.Distinct().OrderBy(c => c).ToArray();
        int classCount = classes.Length;
        var classifiers = new List<BinarySvm>();

        // One-vs-One: 모든 클래스 쌍에 대해 바이너리 SVM 학습
        for (int i = 0; i < classCount; i++)
        {
            for (int j = i + 1; j < classCount; j++)
            {
                int classA = (int)classes[i];
                int classB = (int)classes[j];

                // 해당 클래스 쌍의 데이터만 필터링
                var indices = Enumerable.Range(0, labels.Length)
                    .Where(idx => (int)labels[idx] == classA || (int)labels[idx] == classB)
                    .ToArray();

                var subFeatures = indices.Select(idx => features[idx]).ToArray();
                var subLabels = indices.Select(idx => labels[idx]).ToArray();

                var svm = new BinarySvm(features[0].Length, classA, classB);
                svm.Train(subFeatures, subLabels, _learningRate, _lambda, _epochs);
                classifiers.Add(svm);
            }
        }

        return new SvmModel(classifiers, classCount);
    }

    public double Evaluate(IModel model, double[][] features, double[] labels)
    {
        var predictions = model.Predict(features);
        int correct = predictions.Zip(labels).Count(p => Math.Abs(p.First - p.Second) < 0.5);
        return (double)correct / labels.Length;
    }
}
