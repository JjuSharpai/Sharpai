using SharpAI.Domain.Interfaces;
using SharpAI.Domain.Models;

namespace SharpAI.Infrastructure.Data;

public class CsvDataLoader : IDataLoader
{
    private readonly string _cacheDir;

    public CsvDataLoader(string? cacheDir = null)
    {
        _cacheDir = cacheDir ?? Path.Combine(AppContext.BaseDirectory, "datasets");
        Directory.CreateDirectory(_cacheDir);
    }

    public async Task<DataSet> LoadAsync(string url, Func<string[], (double[] Features, double Label)> parser,
        char separator = ',', bool hasHeader = false)
    {
        var fileName = Path.GetFileName(new Uri(url).AbsolutePath);
        var filePath = Path.Combine(_cacheDir, fileName);

        if (!File.Exists(filePath))
        {
            using var client = new HttpClient();
            var content = await client.GetStringAsync(url);
            await File.WriteAllTextAsync(filePath, content);
        }

        var lines = (await File.ReadAllLinesAsync(filePath))
            .Skip(hasHeader ? 1 : 0)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();

        var features = new double[lines.Length][];
        var labels = new double[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(separator);
            var (f, l) = parser(parts);
            features[i] = f;
            labels[i] = l;
        }

        return new DataSet(features, labels);
    }
}
