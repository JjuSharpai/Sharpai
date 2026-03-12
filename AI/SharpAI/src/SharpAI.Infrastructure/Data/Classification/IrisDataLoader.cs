using SharpAI.Domain.Models;

namespace SharpAI.Infrastructure.Data.Classification;

public class IrisDataLoader : BaseDownloadableDataLoader
{
    public override string Name => "iris";
    protected override string Url => "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
    protected override string FileName => "iris.data";

    private static readonly Dictionary<string, double> LabelMap = new()
    {
        ["Iris-setosa"] = 0,
        ["Iris-versicolor"] = 1,
        ["Iris-virginica"] = 2
    };

    public override async Task<DataSet> LoadAsync()
    {
        await EnsureDownloadedAsync();

        var lines = (await File.ReadAllLinesAsync(FilePath))
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();

        var features = new double[lines.Length][];
        var labels = new double[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            features[i] = [
                double.Parse(parts[0]),
                double.Parse(parts[1]),
                double.Parse(parts[2]),
                double.Parse(parts[3])
            ];
            labels[i] = LabelMap[parts[4].Trim()];
        }

        return new DataSet(features, labels);
    }
}
