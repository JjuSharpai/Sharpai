from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def main():
    # 데이터 로드
    iris = load_iris()
    X, y = iris.data, iris.target

    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # SVM 모델 학습
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_train, y_train)

    # 추론
    y_pred = model.predict(X_test)

    # 결과 출력
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))


if __name__ == "__main__":
    main()
