using SharpAI.Domain.Models;

namespace SharpAI.Infrastructure.Data.Regression;

public class WineQualityDataLoader : BaseDownloadableDataLoader
{
    public override string Name => "wine-quality";
    protected override string Url => "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv";
    protected override string FileName => "winequality-red.csv";

    public override async Task<DataSet> LoadAsync()
    {
        await EnsureDownloadedAsync();

        var lines = (await File.ReadAllLinesAsync(FilePath))
            .Skip(1) // 헤더 스킵
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();

        var features = new double[lines.Length][];
        var labels = new double[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(';');
            // 마지막 컬럼이 quality (라벨), 나머지가 피처
            features[i] = parts[..^1].Select(double.Parse).ToArray();
            labels[i] = double.Parse(parts[^1]);
        }

        return new DataSet(features, labels);
    }
}
