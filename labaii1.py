import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Загружаем датасет
df = pd.read_csv("train.csv")

# 2. Выводим данные
print("Первые 5 строк:")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())

# 3. Проверяем пропущенные значения
print("\nПропущенные значения:")
print(df.isnull().sum())

# 4. Заполняем пропуски

# Age заполняем медианой
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked заполняем модой
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin удаляем из-за большого количества пропусков
df = df.drop("Cabin", axis=1)

print("\nПропуски после заполнения:")
print(df.isnull().sum())

# 5. Нормализация числовых данных
scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)

print("\nДанные после нормализации:")
print(df[["Age", "Fare"]].head())

# 6. Преобразование категориальных данных
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)

print("\nДанные после преобразования категориальных признаков:")
print(df.head())

# Сохраняем обработанный датасет
df.to_csv("processed_train.csv", index=False)

print("\nОбработанный датасет сохранён в processed_train.csv")