using SharpAI.Domain.Enums;

namespace SharpAI.Domain.Interfaces;

public interface IModelStorage
{
    void Save(string path, ModelFormat format = ModelFormat.Json);
    void Load(string path, ModelFormat format = ModelFormat.Json);
}
