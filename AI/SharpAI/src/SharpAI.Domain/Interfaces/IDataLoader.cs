using SharpAI.Domain.Models;

namespace SharpAI.Domain.Interfaces;

public interface IDataLoader
{
    Task<DataSet> LoadAsync(string url, Func<string[], (double[] Features, double Label)> parser,
        char separator = ',', bool hasHeader = false);
}
