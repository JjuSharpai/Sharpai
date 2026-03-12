using SharpAI.Domain.Interfaces;
using SharpAI.Domain.Models;

namespace SharpAI.Infrastructure.Data;

public abstract class BaseDownloadableDataLoader : IDataLoader
{
    public abstract string Name { get; }
    protected abstract string Url { get; }
    protected abstract string FileName { get; }

    protected string DataDir
    {
        get
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "datasets", Name);
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    protected string FilePath => Path.Combine(DataDir, FileName);

    protected async Task EnsureDownloadedAsync()
    {
        if (File.Exists(FilePath)) return;

        using var client = new HttpClient();
        var content = await client.GetStringAsync(Url);
        await File.WriteAllTextAsync(FilePath, content);
    }

    public abstract Task<DataSet> LoadAsync();
}
