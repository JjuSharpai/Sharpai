# Sharpai

## 목표

### ML.NET & 자체 제작 코드
- [ ] ML.NET 기반 모델 학습 및 추론
- [ ] 자체 코드로 ML 알고리즘 구현

### LLM 직접 구현
- [ ] Transformer 아키텍처 직접 구현
- [ ] 토크나이저 구현
- [ ] 학습 파이프라인 구성

### Agent 구축
- [ ] LangChain 기반 Agent 구현
- [ ] Semantic Kernel 기반 Agent 구현

## 실행 방법

### SharpAI (C# file-based app)
```bash
cd AI/SharpAI/samples
dotnet IrisSvm.cs
```

> `.cs` 파일을 직접 실행하려면 아래 명령으로 파일 연결을 설정:
> ```bash
> # CMD
> assoc .cs=DotNetScript
> ftype DotNetScript=dotnet "%1" %*
>
> # PowerShell
> cmd /c 'assoc .cs=DotNetScript'
> cmd /c 'ftype DotNetScript=dotnet "%1" %*'
> ```
> 이후 `IrisSvm.cs`만으로 실행 가능.

### ML.NET
```bash
cd AI/ML.NET/IrisSvm
dotnet run
```

### PyAI (Python)
```bash
cd AI/PyAI
pip install -r requirements.txt
python iris_svm.py
```
