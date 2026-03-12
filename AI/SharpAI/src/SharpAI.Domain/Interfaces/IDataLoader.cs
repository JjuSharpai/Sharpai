using SharpAI.Domain.Models;

namespace SharpAI.Domain.Interfaces;

public interface IDataLoader
{
    Task<DataSet> LoadAsync();
    string Name { get; }
}
