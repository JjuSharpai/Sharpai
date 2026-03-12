using SharpAI.Domain.Interfaces;

namespace SharpAI.Infrastructure.SupervisedLearning;

/// <summary>
/// 멀티클래스 SVM 모델 (One-vs-One 전략).
/// 각 클래스 쌍에 대해 학습된 바이너리 SVM들의 투표로 분류.
/// </summary>
public class SvmModel : IModel
{
    private readonly List<BinarySvm> _classifiers;
    private readonly int _classCount;

    public SvmModel(List<BinarySvm> classifiers, int classCount)
    {
        _classifiers = classifiers;
        _classCount = classCount;
    }

    public double[] Predict(double[][] inputs)
    {
        var results = new double[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = PredictSingle(inputs[i]);
        }
        return results;
    }

    private double PredictSingle(double[] input)
    {
        var votes = new int[_classCount];

        foreach (var svm in _classifiers)
        {
            int predicted = svm.Predict(input);
            votes[predicted]++;
        }

        return Array.IndexOf(votes, votes.Max());
    }
}

/// <summary>
/// 바이너리 Linear SVM (SGD 기반 학습).
/// </summary>
public class BinarySvm
{
    public double[] Weights { get; }
    public double Bias { get; private set; }
    public int ClassA { get; }
    public int ClassB { get; }

    public BinarySvm(int featureCount, int classA, int classB)
    {
        Weights = new double[featureCount];
        Bias = 0;
        ClassA = classA;
        ClassB = classB;
    }

    public void Train(double[][] features, double[] labels, double lr, double lambda, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < features.Length; i++)
            {
                // y = +1 (ClassA), -1 (ClassB)
                double y = (int)labels[i] == ClassA ? 1.0 : -1.0;
                double score = DotProduct(features[i]) + Bias;

                if (y * score < 1)
                {
                    // 오분류 또는 마진 내부: 힌지 로스 그래디언트
                    for (int j = 0; j < Weights.Length; j++)
                        Weights[j] += lr * (y * features[i][j] - lambda * Weights[j]);
                    Bias += lr * y;
                }
                else
                {
                    // 올바르게 분류: 정규화만 적용
                    for (int j = 0; j < Weights.Length; j++)
                        Weights[j] -= lr * lambda * Weights[j];
                }
            }
        }
    }

    public int Predict(double[] input)
    {
        return DotProduct(input) + Bias >= 0 ? ClassA : ClassB;
    }

    private double DotProduct(double[] input)
    {
        double sum = 0;
        for (int i = 0; i < Weights.Length; i++)
            sum += Weights[i] * input[i];
        return sum;
    }
}
