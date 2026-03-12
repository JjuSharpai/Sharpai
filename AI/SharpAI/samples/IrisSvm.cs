#:project ../src/SharpAI.Domain/SharpAI.Domain.csproj
#:project ../src/SharpAI.Application/SharpAI.Application.csproj
#:project ../src/SharpAI.Infrastructure/SharpAI.Infrastructure.csproj

using SharpAI.Application.Services;
using SharpAI.Infrastructure.Data.Classification;
using SharpAI.Infrastructure.SupervisedLearning;

var loader = new IrisDataLoader();
Console.WriteLine("Iris 데이터셋 다운로드 및 로드 중...");
var dataSet = await loader.LoadAsync();

var (trainSet, testSet) = DataSetSplitter.Split(dataSet, testRatio: 0.3, seed: 42);

var learner = new SvmLearner(learningRate: 0.001, lambda: 0.01, epochs: 1000);
var pipeline = new TrainingPipeline();
var result = pipeline.RunSupervised(learner, trainSet, testSet);

Console.WriteLine($"\nAccuracy: {result.Metrics!["accuracy"]:F4}");
Console.WriteLine($"\n예측 결과 (처음 10개):");

string[] classNames = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
for (int i = 0; i < Math.Min(10, result.Predictions.Length); i++)
{
    var actual = classNames[(int)testSet.Labels![i]];
    var predicted = classNames[(int)result.Predictions[i]];
    var mark = actual == predicted ? "O" : "X";
    Console.WriteLine($"  [{mark}] 실제: {actual,-16} 예측: {predicted}");
}
