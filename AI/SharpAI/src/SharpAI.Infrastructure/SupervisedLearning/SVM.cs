using System.Text.Json;
using SharpAI.Domain.Enums;
using SharpAI.Domain.Interfaces;
using SharpAI.Domain.Interfaces.Training;

namespace SharpAI.Infrastructure.SupervisedLearning;

/// <summary>
/// Linear SVM (One-vs-One 멀티클래스 전략, SGD 기반).
/// </summary>
public class SVM : ISupervised, IModelStorage
{
    private readonly double _learningRate;
    private readonly double _lambda;
    private readonly int _epochs;

    private List<BinarySvm> _classifiers = [];
    private int _classCount;

    public SVM(double learningRate = 0.001, double lambda = 0.01, int epochs = 1000)
    {
        _learningRate = learningRate;
        _lambda = lambda;
        _epochs = epochs;
    }

    public void Train(double[][] features, double[] labels)
    {
        var classes = labels.Distinct().OrderBy(c => c).ToArray();
        _classCount = classes.Length;
        _classifiers = [];

        for (int i = 0; i < _classCount; i++)
        {
            for (int j = i + 1; j < _classCount; j++)
            {
                int classA = (int)classes[i];
                int classB = (int)classes[j];

                var indices = Enumerable.Range(0, labels.Length)
                    .Where(idx => (int)labels[idx] == classA || (int)labels[idx] == classB)
                    .ToArray();

                var subFeatures = indices.Select(idx => features[idx]).ToArray();
                var subLabels = indices.Select(idx => labels[idx]).ToArray();

                var svm = new BinarySvm(features[0].Length, classA, classB);
                svm.Train(subFeatures, subLabels, _learningRate, _lambda, _epochs);
                _classifiers.Add(svm);
            }
        }
    }

    public double[] Predict(double[][] inputs)
    {
        var results = new double[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            var votes = new int[_classCount];
            foreach (var svm in _classifiers)
                votes[svm.Predict(inputs[i])]++;
            results[i] = Array.IndexOf(votes, votes.Max());
        }
        return results;
    }

    public double Evaluate(double[][] features, double[] labels)
    {
        var predictions = Predict(features);
        int correct = predictions.Zip(labels).Count(p => Math.Abs(p.First - p.Second) < 0.5);
        return (double)correct / labels.Length;
    }

    public void Save(string path, ModelFormat format = ModelFormat.Json)
    {
        var data = new SvmData
        {
            ClassCount = _classCount,
            Classifiers = _classifiers.Select(c => new BinarySvmData
            {
                Weights = c.Weights,
                Bias = c.Bias,
                ClassA = c.ClassA,
                ClassB = c.ClassB
            }).ToList()
        };

        switch (format)
        {
            case ModelFormat.Json:
                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(path, json);
                break;
            case ModelFormat.Binary:
                using (var stream = File.Create(path))
                using (var writer = new BinaryWriter(stream))
                {
                    writer.Write(data.ClassCount);
                    writer.Write(data.Classifiers.Count);
                    foreach (var c in data.Classifiers)
                    {
                        writer.Write(c.ClassA);
                        writer.Write(c.ClassB);
                        writer.Write(c.Bias);
                        writer.Write(c.Weights.Length);
                        foreach (var w in c.Weights)
                            writer.Write(w);
                    }
                }
                break;
        }
    }

    public void Load(string path, ModelFormat format = ModelFormat.Json)
    {
        SvmData data;

        switch (format)
        {
            case ModelFormat.Json:
                var json = File.ReadAllText(path);
                data = JsonSerializer.Deserialize<SvmData>(json)!;
                break;
            case ModelFormat.Binary:
                using (var stream = File.OpenRead(path))
                using (var reader = new BinaryReader(stream))
                {
                    data = new SvmData
                    {
                        ClassCount = reader.ReadInt32(),
                        Classifiers = []
                    };
                    int count = reader.ReadInt32();
                    for (int i = 0; i < count; i++)
                    {
                        int classA = reader.ReadInt32();
                        int classB = reader.ReadInt32();
                        double bias = reader.ReadDouble();
                        int weightCount = reader.ReadInt32();
                        var weights = new double[weightCount];
                        for (int j = 0; j < weightCount; j++)
                            weights[j] = reader.ReadDouble();
                        data.Classifiers.Add(new BinarySvmData
                        {
                            ClassA = classA, ClassB = classB,
                            Bias = bias, Weights = weights
                        });
                    }
                }
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(format));
        }

        _classCount = data.ClassCount;
        _classifiers = data.Classifiers.Select(c =>
        {
            var svm = new BinarySvm(c.Weights.Length, c.ClassA, c.ClassB);
            svm.LoadWeights(c.Weights, c.Bias);
            return svm;
        }).ToList();
    }

    private class SvmData
    {
        public int ClassCount { get; set; }
        public List<BinarySvmData> Classifiers { get; set; } = [];
    }

    private class BinarySvmData
    {
        public double[] Weights { get; set; } = [];
        public double Bias { get; set; }
        public int ClassA { get; set; }
        public int ClassB { get; set; }
    }

    private class BinarySvm
    {
        private double[] _weights;
        private double _bias;
        private readonly int _classA;
        private readonly int _classB;

        public double[] Weights => _weights;
        public double Bias => _bias;
        public int ClassA => _classA;
        public int ClassB => _classB;

        public BinarySvm(int featureCount, int classA, int classB)
        {
            _weights = new double[featureCount];
            _classA = classA;
            _classB = classB;
        }

        public void LoadWeights(double[] weights, double bias)
        {
            _weights = weights;
            _bias = bias;
        }

        public void Train(double[][] features, double[] labels, double lr, double lambda, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < features.Length; i++)
                {
                    double y = (int)labels[i] == _classA ? 1.0 : -1.0;
                    double score = DotProduct(features[i]) + _bias;

                    if (y * score < 1)
                    {
                        for (int j = 0; j < _weights.Length; j++)
                            _weights[j] += lr * (y * features[i][j] - lambda * _weights[j]);
                        _bias += lr * y;
                    }
                    else
                    {
                        for (int j = 0; j < _weights.Length; j++)
                            _weights[j] -= lr * lambda * _weights[j];
                    }
                }
            }
        }

        public int Predict(double[] input) =>
            DotProduct(input) + _bias >= 0 ? _classA : _classB;

        private double DotProduct(double[] input)
        {
            double sum = 0;
            for (int i = 0; i < _weights.Length; i++)
                sum += _weights[i] * input[i];
            return sum;
        }
    }
}
